"""
Transformer Model for Fingering Correction (Explicit Correction)

Key changes:
- Keep original prediction if correct
- Only correct when original prediction is wrong
- Predict needs_correction flag
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding"""
    
    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class NoteEncoder(nn.Module):
    """Encode individual notes within a group"""
    
    def __init__(self, note_feature_dim: int, d_model: int, max_notes: int = 8):
        super().__init__()
        
        self.note_proj = nn.Linear(note_feature_dim, d_model)
        self.note_pos_embedding = nn.Embedding(max_notes, d_model)
        self.note_norm = nn.LayerNorm(d_model, eps=1e-6)
        
    def forward(
        self,
        note_features: torch.Tensor,
        note_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            note_embeddings: (batch, seq_len, max_notes, d_model)
            group_embeddings: (batch, seq_len, d_model)
        """
        batch_size, seq_len, max_notes, feature_dim = note_features.shape
        
        # Project features
        note_emb = self.note_proj(note_features)
        
        # Add positional embedding
        note_positions = torch.arange(max_notes, device=note_features.device)
        note_pos_emb = self.note_pos_embedding(note_positions)
        note_emb = note_emb + note_pos_emb.unsqueeze(0).unsqueeze(0)
        
        # Apply LayerNorm (skip attention to avoid NaN issues)
        note_embeddings = self.note_norm(note_emb)
        
        # Compute group embeddings as weighted average
        mask_expanded = note_mask.unsqueeze(-1).float()
        group_sum = (note_embeddings * mask_expanded).sum(dim=2)
        group_count = mask_expanded.sum(dim=2).clamp(min=1)
        group_embeddings = group_sum / group_count
        
        return note_embeddings, group_embeddings


class FingeringCorrectionTransformer(nn.Module):
    """
    Transformer for fingering CORRECTION (not just prediction)

    Key idea:
    - Explicitly receives original predictions as input
    - Predicts whether correction is needed (needs_correction)
    - Suggests new fingering only when correction is needed

    Output:
    - needs_correction: (batch, seq, notes) - whether correction is needed
    - corrected_class: (batch, seq, notes, num_classes) - corrected fingering
    """
    
    def __init__(
        self,
        note_feature_dim: int = 77,
        max_notes: int = 8,
        d_model: int = 256,
        nhead: int = 8,
        num_encoder_layers: int = 4,
        num_decoder_layers: int = 4,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        num_classes: int = 11,
    ):
        super().__init__()
        
        self.d_model = d_model
        self.num_classes = num_classes
        self.max_notes = max_notes
        
        # Note encoder
        self.note_encoder = NoteEncoder(note_feature_dim, d_model, max_notes)
        
        # Original prediction embedding (important: explicitly embed original predictions)
        self.original_class_embedding = nn.Embedding(num_classes + 1, d_model // 4)  # +1 for no prediction
        
        # Combine note features with original prediction
        self.combine_proj = nn.Linear(d_model + d_model // 4, d_model)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model, dropout=dropout)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_encoder_layers
        )
        
        # Feature fusion (simplified structure to prevent NaN)
        self.fusion_norm = nn.LayerNorm(d_model, eps=1e-6)
        
        # Prediction heads
        # Head 1: whether correction is needed (needs_correction)
        self.correction_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1)  # Binary: needs correction?
        )
        
        # Head 2: corrected fingering (corrected_class)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(
        self,
        note_features: torch.Tensor,
        note_mask: torch.Tensor,
        seq_mask: torch.Tensor,
        original_classes: torch.Tensor  # (batch, seq, notes) - original prediction classes
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            note_features: (batch, seq_len, max_notes, feature_dim)
            note_mask: (batch, seq_len, max_notes)
            seq_mask: (batch, seq_len)
            original_classes: (batch, seq_len, max_notes) - original predictions (0=none, 1-10=fingering)

        Returns:
            correction_logits: (batch, seq_len, max_notes) - correction probability
            class_logits: (batch, seq_len, max_notes, num_classes) - corrected fingering
        """
        batch_size, seq_len, max_notes, _ = note_features.shape
        
        # Encode notes
        note_emb, group_emb = self.note_encoder(note_features, note_mask)
        
        # Embed original predictions (key: explicitly condition on original predictions)
        # Clamp to valid range
        original_classes_clamped = original_classes.clamp(0, self.num_classes)
        orig_emb = self.original_class_embedding(original_classes_clamped)
        
        # Combine note embeddings with original prediction embeddings
        combined = torch.cat([note_emb, orig_emb], dim=-1)
        note_emb = self.combine_proj(combined)
        
        # Re-aggregate group embeddings after combining
        mask_expanded = note_mask.unsqueeze(-1).float()
        group_sum = (note_emb * mask_expanded).sum(dim=2)
        group_count = mask_expanded.sum(dim=2).clamp(min=1)
        group_emb = group_sum / group_count
        
        # Add positional encoding
        group_emb = self.pos_encoding(group_emb)
        
        # Transformer encoder
        causal_mask = self._generate_causal_mask(seq_len, note_features.device)
        src_key_padding_mask = ~seq_mask
        
        encoded = self.transformer_encoder(
            group_emb,
            mask=causal_mask,
            src_key_padding_mask=src_key_padding_mask
        )
        
        # Feature fusion (simplified: add encoded to note_emb)
        encoded_expanded = encoded.unsqueeze(2).expand(-1, -1, max_notes, -1)
        fused = note_emb + encoded_expanded
        fused = self.fusion_norm(fused)
        
        # Prediction heads
        correction_logits = self.correction_head(fused).squeeze(-1)  # (batch, seq, notes)
        class_logits = self.classifier(fused)  # (batch, seq, notes, num_classes)
        
        return correction_logits, class_logits
    
    def _generate_causal_mask(self, size: int, device: torch.device) -> torch.Tensor:
        mask = torch.triu(torch.ones(size, size, device=device), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask
    
    @torch.no_grad()
    def predict(
        self,
        note_features: torch.Tensor,
        note_mask: torch.Tensor,
        seq_mask: torch.Tensor,
        original_classes: torch.Tensor,
        correction_threshold: float = 0.5
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Predict with correction logic
        
        Returns:
            final_predictions: (batch, seq, notes) - final fingering (keep original or corrected)
            needs_correction: (batch, seq, notes) - whether correction was applied
            correction_probs: (batch, seq, notes) - correction probability (class-based)
        """
        correction_logits, class_logits = self.forward(
            note_features, note_mask, seq_mask, original_classes
        )
        
        # Compute class probabilities
        class_probs = torch.softmax(class_logits, dim=-1)  # (batch, seq, notes, num_classes)
        
        # Probability for the original class
        batch_size, seq_len, max_notes = original_classes.shape
        original_classes_expanded = original_classes.unsqueeze(-1)  # (batch, seq, notes, 1)
        original_class_probs = class_probs.gather(-1, original_classes_expanded).squeeze(-1)  # (batch, seq, notes)
        
        # Top-1 prediction and probability
        top1_probs, top1_preds = class_probs.max(dim=-1)  # (batch, seq, notes)
        
        # Correction decision: only correct when the alternative is clearly better than the original
        # When top1 probability is above threshold and clearly higher than original (e.g., 2x or more)
        # Higher threshold means fewer corrections
        confidence_ratio = top1_probs / (original_class_probs + 1e-6)  # how much better top1 is than original
        needs_correction = (
            (top1_preds != original_classes) &  # prediction differs from original
            (top1_probs > correction_threshold) &  # top1 probability is high enough
            (confidence_ratio > 2.0)  # at least 2x more confident than original
        )
        
        # correction_probs is the probability that correction is needed = 1 - original class probability
        correction_probs = 1.0 - original_class_probs
        
        # Missing notes (original_class=0) always use new prediction
        is_missing = (original_classes == 0)
        
        # Final result: use new prediction if missing or correction needed, otherwise keep original
        final_predictions = torch.where(
            is_missing | needs_correction,
            top1_preds,
            original_classes
        )
        
        return final_predictions, needs_correction, correction_probs


class CorrectionLoss(nn.Module):
    """
    Loss function for correction model

    Two tasks:
    1. Predict whether correction is needed (Binary Cross-Entropy)
    2. Predict corrected fingering (Cross-Entropy, only for wrong predictions)
    """
    
    def __init__(
        self,
        num_classes: int = 11,
        label_smoothing: float = 0.1,
        correction_weight: float = 1.0,
        classification_weight: float = 1.0
    ):
        super().__init__()
        
        self.num_classes = num_classes
        self.correction_weight = correction_weight
        self.classification_weight = classification_weight
        
        self.correction_criterion = nn.BCEWithLogitsLoss(reduction='none')
        self.classification_criterion = nn.CrossEntropyLoss(
            label_smoothing=label_smoothing,
            reduction='none'
        )
    
    def forward(
        self,
        correction_logits: torch.Tensor,
        class_logits: torch.Tensor,
        original_classes: torch.Tensor,
        target_classes: torch.Tensor,
        note_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Args:
            correction_logits: (batch, seq, notes)
            class_logits: (batch, seq, notes, num_classes)
            original_classes: (batch, seq, notes) - original predictions
            target_classes: (batch, seq, notes) - ground truth
            note_mask: (batch, seq, notes)
        
        Returns:
            total_loss: scalar
            metrics: dict
        """
        batch_size, seq_len, max_notes = original_classes.shape
        
        # 1. Compute correction target
        # Correction needed when original differs from ground truth (needs_correction = 1)
        needs_correction_target = (original_classes != target_classes).float()

        # When original is 0 (none), treat as needing correction (not addition)
        # Here, missing originals are also treated as needing correction

        # Correction loss
        correction_loss = self.correction_criterion(
            correction_logits, needs_correction_target
        )
        correction_loss = (correction_loss * note_mask.float()).sum() / note_mask.sum().clamp(min=1)
        
        # 2. Classification loss (applied to all notes)
        # The model should also learn to predict the original when it is correct
        class_logits_flat = class_logits.view(-1, self.num_classes)
        target_flat = target_classes.view(-1)
        note_mask_flat = note_mask.view(-1)
        
        classification_loss = self.classification_criterion(
            class_logits_flat, target_flat
        )
        classification_loss = (classification_loss * note_mask_flat.float()).sum() / note_mask_flat.sum().clamp(min=1)
        
        # Correction mask (for metrics computation)
        correction_mask = needs_correction_target.bool() & note_mask
        
        # Total loss
        total_loss = (self.correction_weight * correction_loss + 
                      self.classification_weight * classification_loss)
        
        # Metrics
        with torch.no_grad():
            # Correction accuracy
            correction_preds = (torch.sigmoid(correction_logits) > 0.5).float()
            correction_acc = ((correction_preds == needs_correction_target) & note_mask).sum().float() / note_mask.sum().clamp(min=1)
            
            # Classification accuracy (all notes)
            class_preds = class_logits.argmax(dim=-1)
            classification_acc = ((class_preds == target_classes) & note_mask).sum().float() / note_mask.sum().clamp(min=1)
            
            # Classification accuracy among those needing correction
            if correction_mask.sum() > 0:
                correction_classification_acc = ((class_preds == target_classes) & correction_mask).sum().float() / correction_mask.sum().clamp(min=1)
            else:
                correction_classification_acc = torch.tensor(1.0)
            
            # Overall accuracy (based on final result)
            final_preds = torch.where(
                correction_preds.bool(),
                class_logits.argmax(dim=-1),
                original_classes
            )
            overall_acc = ((final_preds == target_classes) & note_mask).sum().float() / note_mask.sum().clamp(min=1)
            
            # Correction rate
            correction_rate = needs_correction_target[note_mask].mean()
        
        metrics = {
            'loss': total_loss.item(),
            'correction_loss': correction_loss.item(),
            'classification_loss': classification_loss.item() if isinstance(classification_loss, torch.Tensor) else classification_loss,
            'correction_acc': correction_acc.item(),
            'classification_acc': classification_acc.item(),
            'correction_class_acc': correction_classification_acc.item(),  # accuracy among those needing correction
            'overall_acc': overall_acc.item(),
            'correction_rate': correction_rate.item(),
            'num_samples': note_mask.sum().item()
        }
        
        return total_loss, metrics


def create_model(config) -> FingeringCorrectionTransformer:
    """Create model from config"""
    from .features import FeatureExtractor
    
    feature_extractor = FeatureExtractor(config)
    note_feature_dim = feature_extractor.get_note_feature_dim()
    
    return FingeringCorrectionTransformer(
        note_feature_dim=note_feature_dim,
        max_notes=config.max_notes_per_group,
        d_model=config.d_model,
        nhead=config.nhead,
        num_encoder_layers=config.num_encoder_layers,
        num_decoder_layers=config.num_decoder_layers,
        dim_feedforward=config.dim_feedforward,
        dropout=config.dropout,
        num_classes=config.num_classes,
    )
