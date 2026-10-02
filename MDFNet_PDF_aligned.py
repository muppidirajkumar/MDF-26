# -*- coding: utf-8 -*-
"""
MDF-Net - PDF-aligned multimodal deepfake detection implementation.

Aligned with the supplied MDF-Net paper:
- Four streams: Visual + Audio + Transcript + Metadata
- Visual: Swin Transformer + 3D-CNN -> 512-D
- Audio: Wav2Vec2 + 40-bin MFCC -> 512-D
- Transcript: Whisper-generated transcript + XLM-RoBERTa-large -> 512-D
- Metadata: normalized metadata + MLP -> 512-D
- Reliability-Aware Modality Gate
- CMAF: six pairwise interactions (V-A, V-T, V-M, A-T, A-M, T-M)
- 512-D fused representation
- Classifier: 512 -> 256 -> 2, ReLU + Dropout(0.3)
- AdamW, lr=1e-4, weight_decay=1e-2, cosine annealing
- batch_size=16, epochs=50, early stopping=8
- label smoothing=0.1
- AUC/F1/AP/EER/ECE/MCE/MCC/Balanced Accuracy
- FGSM/PGD visual robustness evaluation
- ACT consistency hook for frequency/visual augmentation

IMPORTANT:
The PDF reports experimental results (98.27% AUC etc.), but this code does
not hard-code those results. They must be obtained by actually training and
evaluating the model on the specified data splits.

Expected labels.csv columns:
video_id, video_path, audio_path, transcript_path, metadata_path, label

Label convention:
0 = Real
1 = Fake

Recommended packages:
pip install torch torchvision torchaudio transformers librosa scikit-learn opencv-python pandas pillow
"""

import os
import json
import math
import random
import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import swin_t, Swin_T_Weights

from sklearn.metrics import (
    roc_auc_score,
    f1_score,
    average_precision_score,
    roc_curve,
    accuracy_score,
    precision_score,
    recall_score,
    matthews_corrcoef,
    balanced_accuracy_score,
)

# Optional heavy dependencies. They are imported explicitly so errors are clear.
try:
    import torchaudio
    from torchaudio.transforms import MFCC
except ImportError as exc:
    raise ImportError(
        "torchaudio is required for Wav2Vec2/MFCC audio processing."
    ) from exc

try:
    from transformers import (
        Wav2Vec2Model,
        Wav2Vec2Processor,
        AutoTokenizer,
        AutoModel,
    )
except ImportError as exc:
    raise ImportError(
        "transformers is required for Wav2Vec2 and XLM-R."
    ) from exc


# ==========================================================
# 1. CONFIGURATION - VALUES TAKEN FROM THE SUPPLIED PDF
# ==========================================================

class Config:
    DATA_ROOT = "MultiDF-23"
    CSV_PATH = os.path.join(DATA_ROOT, "labels.csv")
    CHECKPOINT_DIR = "checkpoints"

    FRAME_COUNT = 16
    IMAGE_SIZE = 224
    SAMPLE_RATE = 16000
    AUDIO_SECONDS = 4
    MAX_AUDIO_SAMPLES = SAMPLE_RATE * AUDIO_SECONDS

    EMBED_DIM = 512
    NUM_CLASSES = 2
    NUM_HEADS = 8

    BATCH_SIZE = 16
    EPOCHS = 50
    EARLY_STOPPING_PATIENCE = 8
    LR = 1e-4
    WEIGHT_DECAY = 1e-2
    DROPOUT = 0.30
    LABEL_SMOOTHING = 0.10

    # The paper describes ACT/penultimate-layer consistency but the exact
    # coefficient is not stated in the supplied PDF. Keep it configurable.
    ACT_WEIGHT = 0.10

    # Paper specifies 40 MFCC filterbanks, 400-sample window, 160-sample hop.
    MFCC_N = 40
    MFCC_WIN_LENGTH = 400
    MFCC_HOP_LENGTH = 160

    # Paper reports FGSM epsilon=8/255 in the abstract.
    FGSM_EPSILON = 8.0 / 255.0
    PGD_EPSILON = 8.0 / 255.0
    PGD_ALPHA = 2.0 / 255.0
    PGD_STEPS = 10

    # Pretrained backbones named in the paper.
    WAV2VEC_NAME = "facebook/wav2vec2-base-960h"
    XLMR_NAME = "xlm-roberta-large"

    SEED = 42


def set_seed(seed=Config.SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ==========================================================
# 2. DATASET
# ==========================================================

class MultiDF23Dataset(Dataset):
    """
    Loads original videos/audio/transcripts/metadata from labels.csv.

    Visual output:
        [3, T, 224, 224]

    Audio output:
        [N] raw waveform at 16 kHz

    Transcript output:
        raw ASR text

    Metadata output:
        [4] normalized metadata vector, matching the original project.
    """

    def __init__(
        self,
        csv_path,
        frame_count=Config.FRAME_COUNT,
        image_size=Config.IMAGE_SIZE,
        train=False,
    ):
        self.df = pd.read_csv(csv_path)
        self.frame_count = frame_count
        self.train = train

        self.visual_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

    def __len__(self):
        return len(self.df)

    @staticmethod
    def _safe_path(value):
        if pd.isna(value):
            return ""
        return str(value)

    def _load_visual_frames(self, video_path):
        frames = []

        if os.path.exists(video_path):
            cap = cv2.VideoCapture(video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

            if total_frames > 0:
                # Uniform temporal sampling.
                indices = np.linspace(
                    0,
                    max(total_frames - 1, 0),
                    self.frame_count,
                    dtype=np.int64,
                )

                for idx in indices:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
                    ret, frame = cap.read()

                    if ret:
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        image = Image.fromarray(frame)
                        frames.append(self.visual_transform(image))

            cap.release()

        # Robust fallback.
        while len(frames) < self.frame_count:
            frames.append(torch.zeros(3, Config.IMAGE_SIZE, Config.IMAGE_SIZE))

        return torch.stack(frames[:self.frame_count], dim=1)

    def _load_audio(self, audio_path):
        if not audio_path or not os.path.exists(audio_path):
            return torch.zeros(Config.MAX_AUDIO_SAMPLES, dtype=torch.float32)

        try:
            waveform, sr = torchaudio.load(audio_path)

            # Convert stereo/multi-channel to mono.
            if waveform.dim() == 2:
                waveform = waveform.mean(dim=0)

            waveform = waveform.float()

            if sr != Config.SAMPLE_RATE:
                waveform = torchaudio.functional.resample(
                    waveform,
                    sr,
                    Config.SAMPLE_RATE,
                )

            waveform = waveform[:Config.MAX_AUDIO_SAMPLES]

            if waveform.numel() < Config.MAX_AUDIO_SAMPLES:
                waveform = F.pad(
                    waveform,
                    (0, Config.MAX_AUDIO_SAMPLES - waveform.numel()),
                )

            # Avoid NaN/Inf entering Wav2Vec2.
            waveform = torch.nan_to_num(waveform)

            return waveform
        except Exception:
            return torch.zeros(Config.MAX_AUDIO_SAMPLES, dtype=torch.float32)

    def _load_transcript(self, transcript_path):
        if transcript_path and os.path.exists(transcript_path):
            try:
                with open(transcript_path, "r", encoding="utf-8") as f:
                    return f.read().strip()
            except Exception:
                pass
        return ""

    def _load_metadata(self, metadata_path):
        """
        The paper states real social-media metadata and an MLP/Transformer
        encoder, but does not define a unique fixed metadata schema.
        The four fields below preserve the user's existing labels.csv
        preparation pipeline.
        """
        if metadata_path and os.path.exists(metadata_path):
            try:
                with open(metadata_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)

                return torch.tensor([
                    float(meta.get("fps", 30.0)) / 60.0,
                    float(meta.get("width", 1920)) / 3840.0,
                    float(meta.get("height", 1080)) / 2160.0,
                    float(meta.get("total_frames", 300)) / 1000.0,
                ], dtype=torch.float32)
            except Exception:
                pass

        return torch.zeros(4, dtype=torch.float32)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        video_path = self._safe_path(row.get("video_path", ""))
        audio_path = self._safe_path(row.get("audio_path", ""))
        transcript_path = self._safe_path(row.get("transcript_path", ""))
        metadata_path = self._safe_path(row.get("metadata_path", ""))

        return {
            "video_id": str(row.get("video_id", idx)),
            "visual": self._load_visual_frames(video_path),
            "audio": self._load_audio(audio_path),
            "transcript": self._load_transcript(transcript_path),
            "metadata": self._load_metadata(metadata_path),
            "label": torch.tensor(int(row["label"]), dtype=torch.long),
        }


# ==========================================================
# 3. VISUAL ENCODER
#    Swin Transformer -> temporal 3D-CNN -> 512-D
# ==========================================================

class VisualEncoder(nn.Module):
    def __init__(self, embed_dim=Config.EMBED_DIM):
        super().__init__()

        # Swin-T provides frame-level spatial representations.
        swin = swin_t(weights=Swin_T_Weights.DEFAULT)

        self.swin_features = swin.features
        self.swin_norm = swin.norm
        self.swin_avgpool = swin.avgpool

        # Swin-T final feature dimension is 768.
        # Input to 3D-CNN: [B, 768, T, 7, 7]
        self.temporal_3d = nn.Sequential(
            nn.Conv3d(
                768,
                256,
                kernel_size=(3, 3, 3),
                padding=1,
                bias=False,
            ),
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True),

            nn.Conv3d(
                256,
                256,
                kernel_size=(3, 3, 3),
                padding=1,
                bias=False,
            ),
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True),

            nn.AdaptiveAvgPool3d((1, 1, 1)),
        )

        self.proj = nn.Sequential(
            nn.Linear(256, embed_dim),
            nn.LayerNorm(embed_dim),
        )

    def forward(self, x):
        # x: [B, 3, T, H, W]
        b, c, t, h, w = x.shape

        # Process every frame with Swin.
        frames = x.permute(0, 2, 1, 3, 4).reshape(
            b * t, c, h, w
        )

        z = self.swin_features(frames)
        z = self.swin_norm(z)

        # Swin features: [BT, H', W', C].
        z = z.permute(0, 3, 1, 2).contiguous()

        _, channels, fh, fw = z.shape
        z = z.reshape(b, t, channels, fh, fw)
        z = z.permute(0, 2, 1, 3, 4).contiguous()

        # Temporal modelling.
        z = self.temporal_3d(z)
        z = z.flatten(1)

        return self.proj(z)


# ==========================================================
# 4. AUDIO ENCODER
#    Wav2Vec2 + 40-MFCC -> 512-D
# ==========================================================

class AudioEncoder(nn.Module):
    def __init__(self, embed_dim=Config.EMBED_DIM):
        super().__init__()

        self.wav2vec = Wav2Vec2Model.from_pretrained(
            Config.WAV2VEC_NAME
        )

        wav_dim = self.wav2vec.config.hidden_size  # normally 768

        self.mfcc = MFCC(
            sample_rate=Config.SAMPLE_RATE,
            n_mfcc=Config.MFCC_N,
            melkwargs={
                "n_fft": 512,
                "win_length": Config.MFCC_WIN_LENGTH,
                "hop_length": Config.MFCC_HOP_LENGTH,
                "n_mels": Config.MFCC_N,
            },
        )

        self.wav_proj = nn.Linear(wav_dim, 256)
        self.mfcc_proj = nn.Linear(Config.MFCC_N, 256)

        self.fusion = nn.Sequential(
            nn.Linear(512, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, waveform):
        # waveform: [B, samples]

        wav_out = self.wav2vec(
            input_values=waveform
        ).last_hidden_state

        wav_feat = wav_out.mean(dim=1)
        wav_feat = self.wav_proj(wav_feat)

        mfcc_feat = self.mfcc(waveform)
        # [B, n_mfcc, time] -> [B, time, n_mfcc]
        mfcc_feat = mfcc_feat.transpose(1, 2)
        mfcc_feat = mfcc_feat.mean(dim=1)
        mfcc_feat = self.mfcc_proj(mfcc_feat)

        fused = torch.cat([wav_feat, mfcc_feat], dim=-1)

        return self.fusion(fused)


# ==========================================================
# 5. TRANSCRIPT ENCODER
#    Whisper-generated transcript -> XLM-R-large -> 512-D
# ==========================================================

class TranscriptEncoder(nn.Module):
    def __init__(self, embed_dim=Config.EMBED_DIM):
        super().__init__()

        self.tokenizer = AutoTokenizer.from_pretrained(
            Config.XLMR_NAME
        )

        self.xlmr = AutoModel.from_pretrained(
            Config.XLMR_NAME
        )

        text_dim = self.xlmr.config.hidden_size  # 1024 for XLM-R-large

        self.proj = nn.Sequential(
            nn.Linear(text_dim, embed_dim),
            nn.LayerNorm(embed_dim),
        )

    def forward(self, texts, device):
        encoded = self.tokenizer(
            list(texts),
            padding=True,
            truncation=True,
            max_length=256,
            return_tensors="pt",
        )

        encoded = {
            k: v.to(device)
            for k, v in encoded.items()
        }

        outputs = self.xlmr(**encoded)
        hidden = outputs.last_hidden_state
        mask = encoded["attention_mask"].unsqueeze(-1).float()

        pooled = (hidden * mask).sum(dim=1)
        pooled = pooled / mask.sum(dim=1).clamp_min(1.0)

        return self.proj(pooled)


# ==========================================================
# 6. METADATA ENCODER
# ==========================================================

class MetadataMLP(nn.Module):
    def __init__(self, in_dim=4, embed_dim=Config.EMBED_DIM):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, embed_dim),
            nn.LayerNorm(embed_dim),
        )

    def forward(self, x):
        return self.mlp(x)


# ==========================================================
# 7. RELIABILITY-AWARE MODALITY GATE
# ==========================================================

class ModalityGate(nn.Module):
    """
    Implements the paper's reliability-conditioned modality weighting:
        r_i = sigmoid(W_r F_i + b_r)
        alpha_i = softmax(r_i)
        Fhat_i = alpha_i * F_i
    """

    def __init__(self, embed_dim=Config.EMBED_DIM):
        super().__init__()

        self.reliability = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embed_dim, 128),
                nn.ReLU(inplace=True),
                nn.Linear(128, 1),
                nn.Sigmoid(),
            )
            for _ in range(4)
        ])

    def forward(self, f_v, f_a, f_t, f_m):
        features = [f_v, f_a, f_t, f_m]

        reliabilities = [
            gate(feat) for gate, feat in zip(self.reliability, features)
        ]

        r = torch.cat(reliabilities, dim=-1)
        alpha = F.softmax(r, dim=-1)

        gated = [
            feat * alpha[:, i:i + 1]
            for i, feat in enumerate(features)
        ]

        return (*gated, alpha, r)


# ==========================================================
# 8. CMAF - SIX PAIRWISE CROSS-MODAL INTERACTIONS
# ==========================================================

class SinglePairCrossAttention(nn.Module):
    def __init__(
        self,
        embed_dim=Config.EMBED_DIM,
        num_heads=Config.NUM_HEADS,
    ):
        super().__init__()

        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            batch_first=True,
        )

        self.norm1 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Linear(embed_dim * 2, embed_dim),
        )
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, q_feat, kv_feat):
        q = q_feat.unsqueeze(1)
        kv = kv_feat.unsqueeze(1)

        attn_out, _ = self.attn(
            query=q,
            key=kv,
            value=kv,
            need_weights=False,
        )

        x = self.norm1(q + attn_out)
        x = self.norm2(x + self.ffn(x))

        return x.squeeze(1)


class CMAFModule(nn.Module):
    """
    Six C(4,2)=6 pairwise interactions:
        V-A, V-T, V-M, A-T, A-M, T-M
    """

    def __init__(self, embed_dim=Config.EMBED_DIM):
        super().__init__()

        self.va = SinglePairCrossAttention(embed_dim)
        self.vt = SinglePairCrossAttention(embed_dim)
        self.vm = SinglePairCrossAttention(embed_dim)
        self.at = SinglePairCrossAttention(embed_dim)
        self.am = SinglePairCrossAttention(embed_dim)
        self.tm = SinglePairCrossAttention(embed_dim)

        # Six 512-D outputs -> one 512-D fused representation.
        self.fusion = nn.Sequential(
            nn.Linear(embed_dim * 6, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(Config.DROPOUT),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.LayerNorm(embed_dim),
        )

    def forward(self, f_v, f_a, f_t, f_m):
        pair_features = [
            self.va(f_v, f_a),
            self.vt(f_v, f_t),
            self.vm(f_v, f_m),
            self.at(f_a, f_t),
            self.am(f_a, f_m),
            self.tm(f_t, f_m),
        ]

        pair_concat = torch.cat(pair_features, dim=-1)

        return self.fusion(pair_concat)


# ==========================================================
# 9. COMPLETE MDF-NET
# ==========================================================

class MDFNet(nn.Module):
    def __init__(self, freeze_text_backbone=False):
        super().__init__()

        self.visual_enc = VisualEncoder()
        self.audio_enc = AudioEncoder()
        self.text_enc = TranscriptEncoder()
        self.metadata_enc = MetadataMLP()

        if freeze_text_backbone:
            for p in self.text_enc.xlmr.parameters():
                p.requires_grad = False

        self.modality_gate = ModalityGate()
        self.cmaf = CMAFModule()

        # PDF: 512 -> 256 -> 2, ReLU + Dropout(0.3)
        self.classifier = nn.Sequential(
            nn.Linear(Config.EMBED_DIM, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(Config.DROPOUT),
            nn.Linear(256, Config.NUM_CLASSES),
        )

    def forward(self, visual, audio, transcript, metadata):
        f_v = self.visual_enc(visual)
        f_a = self.audio_enc(audio)
        f_t = self.text_enc(transcript, visual.device)
        f_m = self.metadata_enc(metadata)

        (
            f_vg,
            f_ag,
            f_tg,
            f_mg,
            gate_weights,
            reliabilities,
        ) = self.modality_gate(f_v, f_a, f_t, f_m)

        fused = self.cmaf(f_vg, f_ag, f_tg, f_mg)

        logits = self.classifier(fused)
        probs = F.softmax(logits, dim=-1)

        return {
            "logits": logits,
            "probs": probs,
            "fused": fused,
            "visual_feat": f_v,
            "audio_feat": f_a,
            "text_feat": f_t,
            "metadata_feat": f_m,
            "gate_weights": gate_weights,
            "reliabilities": reliabilities,
        }


# ==========================================================
# 10. ACT - ADVERSARIAL CONSISTENCY TRAINING
# ==========================================================

def frequency_domain_augmentation(
    visual,
    gaussian_std=0.01,
    jpeg_like_strength=0.0,
    dct_strength=0.02,
):
    """
    Practical frequency-aware augmentation for ACT.

    The supplied paper specifies JPEG compression, Gaussian noise and
    DCT/frequency perturbation, but does not give every augmentation
    coefficient in the PDF. Therefore the coefficients are configurable.
    """

    x = visual

    # Gaussian perturbation.
    if gaussian_std > 0:
        x = x + gaussian_std * torch.randn_like(x)

    # Simple DCT-inspired high-frequency perturbation.
    # We operate on the spatial tensor without adding external dependencies.
    if dct_strength > 0:
        shifted = torch.roll(x, shifts=1, dims=-1)
        high_freq = shifted - x
        x = x + dct_strength * high_freq

    return x


class MDFNetLoss(nn.Module):
    """
    Main paper objective:
        Cross-Entropy with label smoothing
        + ACT penultimate-layer consistency

    ACT coefficient is configurable because the supplied PDF does not
    state one unique numerical value.
    """

    def __init__(
        self,
        label_smoothing=Config.LABEL_SMOOTHING,
        act_weight=Config.ACT_WEIGHT,
    ):
        super().__init__()

        self.ce = nn.CrossEntropyLoss(
            label_smoothing=label_smoothing
        )
        self.act_weight = act_weight

    def forward(
        self,
        clean_logits,
        clean_fused,
        aug_fused,
        targets,
    ):
        ce_loss = self.ce(clean_logits, targets)

        if aug_fused is not None:
            clean_norm = F.normalize(clean_fused, dim=-1)
            aug_norm = F.normalize(aug_fused, dim=-1)

            consistency = (
                1.0 - (clean_norm * aug_norm).sum(dim=-1)
            ).mean()
        else:
            consistency = torch.zeros(
                (),
                device=clean_logits.device,
            )

        total = ce_loss + self.act_weight * consistency

        return total, {
            "ce": float(ce_loss.detach().cpu()),
            "act": float(
                (self.act_weight * consistency).detach().cpu()
            ),
        }


# ==========================================================
# 11. CALIBRATION AND METRICS
# ==========================================================

def expected_calibration_error(probs, labels, num_bins=10):
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    correctness = (predictions == labels).astype(np.float32)

    boundaries = np.linspace(0.0, 1.0, num_bins + 1)
    ece = 0.0
    mce = 0.0

    for i in range(num_bins):
        lower = boundaries[i]
        upper = boundaries[i + 1]

        if i == 0:
            mask = (
                (confidences >= lower)
                & (confidences <= upper)
            )
        else:
            mask = (
                (confidences > lower)
                & (confidences <= upper)
            )

        if not np.any(mask):
            continue

        confidence = confidences[mask].mean()
        accuracy = correctness[mask].mean()
        gap = abs(accuracy - confidence)

        ece += gap * mask.mean()
        mce = max(mce, gap)

    return float(ece), float(mce)


def equal_error_rate(y_true, fake_scores):
    fpr, tpr, _ = roc_curve(y_true, fake_scores)
    fnr = 1.0 - tpr
    index = np.nanargmin(np.abs(fpr - fnr))
    return float(fpr[index])


def calculate_metrics(y_true, probs):
    fake_scores = probs[:, 1]
    preds = (fake_scores >= 0.5).astype(int)

    result = {
        "Accuracy": accuracy_score(y_true, preds),
        "Precision": precision_score(
            y_true, preds, zero_division=0
        ),
        "Recall": recall_score(
            y_true, preds, zero_division=0
        ),
        "F1": f1_score(
            y_true, preds, zero_division=0
        ),
        "AUC": roc_auc_score(y_true, fake_scores),
        "AP": average_precision_score(y_true, fake_scores),
        "EER": equal_error_rate(y_true, fake_scores),
        "MCC": matthews_corrcoef(y_true, preds),
        "BalancedAccuracy": balanced_accuracy_score(
            y_true, preds
        ),
    }

    result["ECE"], result["MCE"] = expected_calibration_error(
        probs,
        y_true,
        num_bins=10,
    )

    return result


# ==========================================================
# 12. TRAIN / VALIDATION
# ==========================================================

def move_batch(batch, device):
    visual = batch["visual"].to(device, non_blocking=True)
    audio = batch["audio"].to(device, non_blocking=True)
    metadata = batch["metadata"].to(device, non_blocking=True)
    labels = batch["label"].to(device, non_blocking=True)

    return visual, audio, batch["transcript"], metadata, labels


def train_one_epoch(model, loader, optimizer, loss_fn, device):
    model.train()

    running_loss = 0.0
    count = 0

    for batch in loader:
        visual, audio, text, metadata, labels = move_batch(
            batch, device
        )

        optimizer.zero_grad(set_to_none=True)

        clean_out = model(
            visual,
            audio,
            text,
            metadata,
        )

        # ACT augmentation is applied to the visual stream.
        augmented_visual = frequency_domain_augmentation(
            visual
        )

        aug_out = model(
            augmented_visual,
            audio,
            text,
            metadata,
        )

        loss, loss_parts = loss_fn(
            clean_logits=clean_out["logits"],
            clean_fused=clean_out["fused"],
            aug_fused=aug_out["fused"],
            targets=labels,
        )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=5.0,
        )

        optimizer.step()

        batch_size = labels.size(0)
        running_loss += loss.item() * batch_size
        count += batch_size

    return running_loss / max(count, 1)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()

    all_labels = []
    all_probs = []
    total_loss = 0.0
    count = 0

    ce = nn.CrossEntropyLoss()

    for batch in loader:
        visual, audio, text, metadata, labels = move_batch(
            batch, device
        )

        out = model(
            visual,
            audio,
            text,
            metadata,
        )

        loss = ce(out["logits"], labels)

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        count += batch_size

        all_labels.append(labels.cpu().numpy())
        all_probs.append(out["probs"].cpu().numpy())

    y_true = np.concatenate(all_labels)
    probs = np.concatenate(all_probs)

    metrics = calculate_metrics(y_true, probs)
    metrics["Loss"] = total_loss / max(count, 1)

    return metrics, y_true, probs


# ==========================================================
# 13. CHECKPOINTING / EARLY STOPPING
# ==========================================================

def save_checkpoint(model, optimizer, scheduler, epoch, metrics, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "metrics": metrics,
        },
        path,
    )


# ==========================================================
# 14. FGSM / PGD VISUAL ROBUSTNESS
# ==========================================================

def fgsm_attack(
    model,
    visual,
    audio,
    text,
    metadata,
    labels,
    epsilon=Config.FGSM_EPSILON,
):
    """
    FGSM is applied to the visual input only, as described in the paper.

    Since the visual tensor is ImageNet-normalized, epsilon is converted
    from pixel scale to normalized feature scale.
    """

    model.eval()

    x = visual.detach().clone().requires_grad_(True)

    out = model(x, audio, text, metadata)
    loss = F.cross_entropy(out["logits"], labels)

    model.zero_grad(set_to_none=True)
    loss.backward()

    grad = x.grad.sign()

    std = torch.tensor(
        [0.229, 0.224, 0.225],
        device=x.device,
    ).view(1, 3, 1, 1, 1)

    eps_norm = epsilon / std

    adv = x + eps_norm * grad

    return adv.detach()


def pgd_attack(
    model,
    visual,
    audio,
    text,
    metadata,
    labels,
    epsilon=Config.PGD_EPSILON,
    alpha=Config.PGD_ALPHA,
    steps=Config.PGD_STEPS,
):
    model.eval()

    original = visual.detach()

    std = torch.tensor(
        [0.229, 0.224, 0.225],
        device=visual.device,
    ).view(1, 3, 1, 1, 1)

    eps_norm = epsilon / std
    alpha_norm = alpha / std

    adv = original.clone()

    for _ in range(steps):
        adv.requires_grad_(True)

        out = model(
            adv,
            audio,
            text,
            metadata,
        )

        loss = F.cross_entropy(
            out["logits"],
            labels,
        )

        model.zero_grad(set_to_none=True)
        loss.backward()

        with torch.no_grad():
            adv = adv + alpha_norm * adv.grad.sign()

            delta = torch.clamp(
                adv - original,
                -eps_norm,
                eps_norm,
            )

            adv = original + delta

    return adv.detach()


def adversarial_evaluation(model, loader, device):
    model.eval()

    y_true = []
    clean_probs = []
    fgsm_probs = []
    pgd_probs = []

    for batch in loader:
        visual, audio, text, metadata, labels = move_batch(
            batch, device
        )

        with torch.no_grad():
            clean = model(
                visual,
                audio,
                text,
                metadata,
            )

        # Attacks need gradients, so they are deliberately executed
        # outside the no-grad context.
        y_true.append(labels.cpu().numpy())
        clean_probs.append(clean["probs"].cpu().numpy())

        fgsm = fgsm_attack(
            model,
            visual,
            audio,
            text,
            metadata,
            labels,
        )

        pgd = pgd_attack(
            model,
            visual,
            audio,
            text,
            metadata,
            labels,
        )

        with torch.no_grad():
            fgsm_out = model(
                fgsm,
                audio,
                text,
                metadata,
            )
            pgd_out = model(
                pgd,
                audio,
                text,
                metadata,
            )

        fgsm_probs.append(fgsm_out["probs"].cpu().numpy())
        pgd_probs.append(pgd_out["probs"].cpu().numpy())

        # Keep evaluation manageable for large datasets.
        if len(y_true) >= 20:
            break

    y = np.concatenate(y_true)
    clean_p = np.concatenate(clean_probs)
    fgsm_p = np.concatenate(fgsm_probs)
    pgd_p = np.concatenate(pgd_probs)

    return {
        "clean": calculate_metrics(y, clean_p),
        "fgsm": calculate_metrics(y, fgsm_p),
        "pgd": calculate_metrics(y, pgd_p),
    }


# ==========================================================
# 15. TRAINING ENTRY POINT
# ==========================================================

def run_training(csv_path=Config.CSV_PATH):
    set_seed()

    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"labels.csv not found: {csv_path}"
        )

    # The CSV should already contain the desired train/validation/test
    # split. If a split column exists, use it. Otherwise use an 80/20 split.
    full_df = pd.read_csv(csv_path)

    if "split" in full_df.columns:
        train_df = full_df[
            full_df["split"].astype(str).str.lower() == "train"
        ].copy()

        val_df = full_df[
            full_df["split"].astype(str).str.lower().isin(
                ["val", "validation"]
            )
        ].copy()

        test_df = full_df[
            full_df["split"].astype(str).str.lower() == "test"
        ].copy()

        if len(val_df) == 0:
            val_df = test_df.copy()

    else:
        # Fallback only. For the paper's symmetric protocol, use the
        # official/verified split information if it exists in your CSV.
        shuffled = full_df.sample(
            frac=1.0,
            random_state=Config.SEED,
        ).reset_index(drop=True)

        n_train = int(0.80 * len(shuffled))
        n_val = int(0.10 * len(shuffled))

        train_df = shuffled.iloc[:n_train]
        val_df = shuffled.iloc[
            n_train:n_train + n_val
        ]
        test_df = shuffled.iloc[
            n_train + n_val:
        ]

    temp_dir = Path(Config.DATA_ROOT) / "_splits"
    temp_dir.mkdir(parents=True, exist_ok=True)

    train_csv = temp_dir / "train.csv"
    val_csv = temp_dir / "val.csv"
    test_csv = temp_dir / "test.csv"

    train_df.to_csv(train_csv, index=False)
    val_df.to_csv(val_csv, index=False)
    test_df.to_csv(test_csv, index=False)

    train_set = MultiDF23Dataset(
        str(train_csv),
        train=True,
    )
    val_set = MultiDF23Dataset(
        str(val_csv),
        train=False,
    )
    test_set = MultiDF23Dataset(
        str(test_csv),
        train=False,
    )

    train_loader = DataLoader(
        train_set,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
    )

    val_loader = DataLoader(
        val_set,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
    )

    test_loader = DataLoader(
        test_set,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
    )

    print(f"Device: {DEVICE}")
    print(f"Train: {len(train_set)}")
    print(f"Validation: {len(val_set)}")
    print(f"Test: {len(test_set)}")

    model = MDFNet().to(DEVICE)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=Config.LR,
        weight_decay=Config.WEIGHT_DECAY,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=Config.EPOCHS,
    )

    loss_fn = MDFNetLoss()

    best_f1 = -1.0
    patience_counter = 0

    for epoch in range(1, Config.EPOCHS + 1):
        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            loss_fn,
            DEVICE,
        )

        val_metrics, _, _ = evaluate(
            model,
            val_loader,
            DEVICE,
        )

        scheduler.step()

        print(
            f"Epoch {epoch:03d}/{Config.EPOCHS} | "
            f"Train Loss={train_loss:.4f} | "
            f"Val AUC={val_metrics['AUC']:.4f} | "
            f"Val F1={val_metrics['F1']:.4f} | "
            f"Val ECE={val_metrics['ECE']:.4f}"
        )

        if val_metrics["F1"] > best_f1:
            best_f1 = val_metrics["F1"]
            patience_counter = 0

            save_checkpoint(
                model,
                optimizer,
                scheduler,
                epoch,
                val_metrics,
                os.path.join(
                    Config.CHECKPOINT_DIR,
                    "mdfnet_best.pt",
                ),
            )
        else:
            patience_counter += 1

        if patience_counter >= Config.EARLY_STOPPING_PATIENCE:
            print("Early stopping triggered.")
            break

    # Load best model.
    checkpoint_path = os.path.join(
        Config.CHECKPOINT_DIR,
        "mdfnet_best.pt",
    )

    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(
            checkpoint_path,
            map_location=DEVICE,
        )
        model.load_state_dict(
            checkpoint["model_state_dict"]
        )

    print("\n=== FINAL TEST RESULTS ===")

    test_metrics, _, _ = evaluate(
        model,
        test_loader,
        DEVICE,
    )

    for key, value in test_metrics.items():
        print(f"{key:20s}: {value:.6f}")

    print("\n=== VISUAL ADVERSARIAL EVALUATION ===")
    # The attack functions intentionally use gradients, so this call
    # should not be wrapped in torch.no_grad().
    adv_results = adversarial_evaluation(
        model,
        test_loader,
        DEVICE,
    )

    for attack_name, metrics in adv_results.items():
        print(f"\n[{attack_name.upper()}]")
        for key in ["AUC", "F1", "AP", "EER", "ECE", "MCE"]:
            print(
                f"{key:10s}: {metrics[key]:.6f}"
            )

    return model, test_metrics, adv_results


# ==========================================================
# 16. FORWARD-PASS VERIFICATION
# ==========================================================

def verify_model(csv_path=Config.CSV_PATH):
    set_seed()

    if not os.path.exists(csv_path):
        print(
            f"[!] labels.csv not found: {csv_path}\n"
            "Run your MultiDF-23 preparation script first."
        )
        return

    dataset = MultiDF23Dataset(
        csv_path,
        frame_count=Config.FRAME_COUNT,
    )

    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
    )

    model = MDFNet().to(DEVICE)
    model.eval()

    batch = next(iter(loader))

    visual, audio, text, metadata, labels = move_batch(
        batch,
        DEVICE,
    )

    with torch.no_grad():
        output = model(
            visual,
            audio,
            text,
            metadata,
        )

    print("\n=== MDF-Net PDF-ALIGNED FORWARD PASS ===")
    print("Visual input       :", tuple(visual.shape))
    print("Audio input        :", tuple(audio.shape))
    print("Metadata input     :", tuple(metadata.shape))
    print("Fused feature      :", tuple(output["fused"].shape))
    print("Logits             :", tuple(output["logits"].shape))
    print("Probabilities      :", output["probs"].cpu().numpy())
    print("Modality weights   :", output["gate_weights"].cpu().numpy())
    print("Reliability scores :", output["reliabilities"].cpu().numpy())
    print("Ground truth       :", labels.cpu().numpy())


# ==========================================================
# 17. COMMAND LINE
# ==========================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="PDF-aligned MDF-Net"
    )

    parser.add_argument(
        "--csv",
        default=Config.CSV_PATH,
        help="Path to labels.csv",
    )

    parser.add_argument(
        "--verify",
        action="store_true",
        help="Run one-batch architecture verification.",
    )

    parser.add_argument(
        "--train",
        action="store_true",
        help="Train MDF-Net.",
    )

    args = parser.parse_args()

    if args.verify:
        verify_model(args.csv)
    elif args.train:
        run_training(args.csv)
    else:
        verify_model(args.csv)
