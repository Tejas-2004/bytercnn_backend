import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import random

# 75 class labels in training data order (from backend/classes.json key "1")
CLASS_LABELS = [
    "JPG", "ARW", "CR2", "DNG", "GPR", "NEF", "NRW", "ORF", "PEF", "RAF",
    "RW2", "3FR", "TIFF", "HEIC", "BMP", "GIF", "PNG", "AI", "EPS", "PSD",
    "MOV", "MP4", "3GP", "AVI", "MKV", "OGV", "WEBM", "APK", "JAR", "MSI",
    "DMG", "7Z", "BZ2", "DEB", "GZ", "PKG", "RAR", "RPM", "XZ", "ZIP",
    "EXE", "MACH-O", "ELF", "DLL", "DOC", "DOCX", "KEY", "PPT", "PPTX",
    "XLS", "XLSX", "DJVU", "EPUB", "MOBI", "PDF", "MD", "RTF", "TXT",
    "TEX", "JSON", "HTML", "XML", "LOG", "CSV", "AIFF", "FLAC", "M4A",
    "MP3", "OGG", "WAV", "WMA", "PCAP", "TTF", "DWG", "SQLITE"
]

# Build reverse lookup: label name -> index in CLASS_LABELS
_LABEL_TO_IDX = {lbl: i for i, lbl in enumerate(CLASS_LABELS)}

CATEGORY_MAP = {
    "ARW": "Raw", "CR2": "Raw", "DNG": "Raw", "GPR": "Raw", "NEF": "Raw",
    "NRW": "Raw", "ORF": "Raw", "PEF": "Raw", "RAF": "Raw", "RW2": "Raw",
    "3FR": "Raw", "JPG": "Bitmap", "TIFF": "Bitmap", "HEIC": "Bitmap",
    "BMP": "Bitmap", "GIF": "Bitmap", "PNG": "Bitmap", "AI": "Vector",
    "EPS": "Vector", "PSD": "Vector", "MOV": "Video", "MP4": "Video",
    "3GP": "Video", "AVI": "Video", "MKV": "Video", "OGV": "Video",
    "WEBM": "Video", "APK": "Archive", "JAR": "Archive", "MSI": "Archive",
    "DMG": "Archive", "7Z": "Archive", "BZ2": "Archive", "DEB": "Archive",
    "GZ": "Archive", "PKG": "Archive", "RAR": "Archive", "RPM": "Archive",
    "XZ": "Archive", "ZIP": "Archive", "EXE": "Executables", "MACH-O": "Executables",
    "ELF": "Executables", "DLL": "Executables", "DOC": "Office", "DOCX": "Office",
    "KEY": "Office", "PPT": "Office", "PPTX": "Office", "XLS": "Office",
    "XLSX": "Office", "DJVU": "Published", "EPUB": "Published", "MOBI": "Published",
    "PDF": "Published", "MD": "Human-readable", "RTF": "Human-readable",
    "TXT": "Human-readable", "TEX": "Human-readable", "JSON": "Human-readable",
    "HTML": "Human-readable", "XML": "Human-readable", "LOG": "Human-readable",
    "CSV": "Human-readable", "AIFF": "Audio", "FLAC": "Audio", "M4A": "Audio",
    "MP3": "Audio", "OGG": "Audio", "WAV": "Audio", "WMA": "Audio",
    "PCAP": "Other", "TTF": "Other", "DWG": "Other", "SQLITE": "Other"
}

NUM_CLASSES = 75
BLOCK_SIZE = 512

# ── File signature (magic byte) detection ──────────────────────
# Maps (offset, magic_bytes) → label. Checked in order; first match wins.
_SIGNATURES: list[tuple[int, bytes, str]] = [
    # Images
    (0, b'\x89PNG\r\n\x1a\n', "PNG"),
    (0, b'\xff\xd8\xff', "JPG"),
    (0, b'GIF87a', "GIF"),
    (0, b'GIF89a', "GIF"),
    (0, b'BM', "BMP"),
    (0, b'\x00\x00\x01\x00', "BMP"),      # ICO shares BMP label
    (0, b'II\x2a\x00', "TIFF"),            # little-endian TIFF
    (0, b'MM\x00\x2a', "TIFF"),            # big-endian TIFF
    # HEIC (ftyp box)
    (4, b'ftyp', "HEIC"),
    # RAW camera
    (0, b'II\x55\x00', "ARW"),             # Sony ARW
    # Video / ftyp-based — refine after HEIC
    (4, b'ftypmp4', "MP4"),
    (4, b'ftypisom', "MP4"),
    (4, b'ftypMSNV', "MP4"),
    (4, b'ftypqt', "MOV"),
    (4, b'ftyp3gp', "3GP"),
    (0, b'\x1a\x45\xdf\xa3', "MKV"),       # also WEBM
    (0, b'RIFF', "AVI"),                    # could be WAV too, refined below
    (0, b'\x00\x00\x00\x1c\x66\x74\x79\x70', "MP4"),
    # Audio
    (0, b'FORM', "AIFF"),
    (0, b'fLaC', "FLAC"),
    (0, b'\xff\xfb', "MP3"),
    (0, b'\xff\xf3', "MP3"),
    (0, b'\xff\xf2', "MP3"),
    (0, b'ID3', "MP3"),
    (0, b'OggS', "OGG"),
    # Archives
    (0, b'PK\x03\x04', "ZIP"),
    (0, b'Rar!\x1a\x07', "RAR"),
    (0, b'\x1f\x8b', "GZ"),
    (0, b'7z\xbc\xaf\x27\x1c', "7Z"),
    (0, b'BZh', "BZ2"),
    (0, b'\xfd7zXZ\x00', "XZ"),
    # Executables
    (0, b'MZ', "EXE"),
    (0, b'\x7fELF', "ELF"),
    (0, b'\xfe\xed\xfa', "MACH-O"),
    (0, b'\xca\xfe\xba\xbe', "MACH-O"),
    # Office / compound docs
    (0, b'\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1', "DOC"),  # OLE2 (DOC/XLS/PPT)
    # PDF
    (0, b'%PDF', "PDF"),
    # Published
    (0, b'AT&TFORM', "DJVU"),
    # Fonts
    (0, b'\x00\x01\x00\x00', "TTF"),
    # Database
    (0, b'SQLite format 3', "SQLITE"),
    # PCAP
    (0, b'\xd4\xc3\xb2\xa1', "PCAP"),
    (0, b'\xa1\xb2\xc3\xd4', "PCAP"),
    # Human-readable (checked via content heuristics below)
]


def detect_signature(file_bytes: bytes) -> str | None:
    """Detect file type from magic bytes. Returns label or None."""
    for offset, magic, label in _SIGNATURES:
        end = offset + len(magic)
        if len(file_bytes) >= end and file_bytes[offset:end] == magic:
            # Refine RIFF: could be AVI or WAV
            if magic == b'RIFF' and len(file_bytes) >= 12:
                sub = file_bytes[8:12]
                if sub == b'AVI ':
                    return "AVI"
                if sub == b'WAVE':
                    return "WAV"
                return "AVI"  # default RIFF → AVI
            # Refine ZIP-based formats by scanning internal entry names.
            # ZIP local-file headers repeat throughout the file; scan a
            # generous window so we catch entries like ppt/, word/, xl/
            # even when [Content_Types].xml is the first entry.
            if label == "ZIP" and len(file_bytes) > 30:
                window = file_bytes[:min(len(file_bytes), 8192)]
                if b'AndroidManifest' in window or b'classes.dex' in window:
                    return "APK"
                if b'word/' in window or b'word\\' in window:
                    return "DOCX"
                if b'xl/' in window or b'xl\\' in window:
                    return "XLSX"
                if b'ppt/' in window or b'ppt\\' in window:
                    return "PPTX"
                if b'META-INF/' in window and b'.class' in window:
                    return "JAR"
                if b'mimetype' in window and b'epub' in window:
                    return "EPUB"
            # Refine OLE2: DOC vs PPT vs XLS
            if magic == b'\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1':
                # Simple heuristic based on common internal markers
                header = file_bytes[:4096] if len(file_bytes) >= 4096 else file_bytes
                if b'PowerPoint' in header or b'P\x00o\x00w\x00e\x00r' in header:
                    return "PPT"
                if b'Excel' in header or b'Workbook' in header:
                    return "XLS"
                return "DOC"
            # Refine ftyp as HEIC vs MP4/MOV
            if magic == b'ftyp':
                ftyp_brand = file_bytes[4:12]
                if b'heic' in ftyp_brand or b'heix' in ftyp_brand or b'mif1' in ftyp_brand:
                    return "HEIC"
                if b'mp4' in ftyp_brand or b'isom' in ftyp_brand:
                    return "MP4"
                if b'qt' in ftyp_brand:
                    return "MOV"
                if b'3gp' in ftyp_brand:
                    return "3GP"
                return "MP4"  # default ftyp
            return label

    # Text-based heuristics (no binary magic)
    head = file_bytes[:1024]
    try:
        text = head.decode('utf-8', errors='strict')
    except UnicodeDecodeError:
        return None
    text_stripped = text.lstrip()
    if text_stripped.startswith('{') or text_stripped.startswith('['):
        return "JSON"
    if text_stripped.startswith('<!') or text_stripped.lower().startswith('<html'):
        return "HTML"
    if text_stripped.startswith('<?xml') or text_stripped.startswith('<'):
        return "XML"
    if text_stripped.startswith('{\\rtf'):
        return "RTF"
    return None


class EfficientByteRCNN(nn.Module):
    def __init__(self, num_classes=75, embed_dim=64, gru_hidden=192, gru_layers=2):
        super().__init__()
        self.embedding = nn.Embedding(256, embed_dim)
        self.embed_dropout = nn.Dropout(0.10)

        self.gru = nn.GRU(embed_dim, gru_hidden, num_layers=gru_layers,
                          bidirectional=True, batch_first=True)

        concat_dim = embed_dim + gru_hidden * 2  # 64 + 384 = 448
        self.bn_input = nn.BatchNorm1d(concat_dim)

        # 5 CNN branches with different kernel sizes
        kernel_sizes = [5, 9, 27, 40, 65]
        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(concat_dim, 128, k, padding=k // 2),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.Conv1d(128, 96, 3, padding=1),
                nn.BatchNorm1d(96),
                nn.ReLU(),
            ) for k in kernel_sizes
        ])
        # Each branch outputs 2*96 = 192, total = 5*192 = 960

        self.fc1 = nn.Linear(960, 1536)
        self.bn_fc1 = nn.BatchNorm1d(1536)
        self.fc2 = nn.Linear(1536, 1024)
        self.bn_fc2 = nn.BatchNorm1d(1024)
        self.fc3 = nn.Linear(1024, 512)
        self.bn_fc3 = nn.BatchNorm1d(512)
        self.fc4 = nn.Linear(512, num_classes)

    def forward(self, x):
        # x: (B, 512) long tensor of byte values
        emb = self.embed_dropout(self.embedding(x))  # (B, 512, 64)
        gru_out, _ = self.gru(emb)  # (B, 512, 384)
        combined = torch.cat([emb, gru_out], dim=2)  # (B, 512, 448)
        combined = combined.permute(0, 2, 1)  # (B, 448, 512)
        combined = self.bn_input(combined)

        branch_outputs = []
        for conv in self.convs:
            out = conv(combined)  # (B, 96, 512)
            max_pool = F.adaptive_max_pool1d(out, 1)
            avg_pool = F.adaptive_avg_pool1d(out, 1)
            branch_outputs.append(torch.cat([max_pool, avg_pool], dim=1).squeeze(-1))

        x = torch.cat(branch_outputs, dim=1)  # (B, 960)
        x = F.relu(self.bn_fc1(self.fc1(x)))
        x = F.dropout(x, p=0.20, training=self.training)
        x = F.relu(self.bn_fc2(self.fc2(x)))
        x = F.dropout(x, p=0.20, training=self.training)
        x = F.relu(self.bn_fc3(self.fc3(x)))
        x = F.dropout(x, p=0.20, training=self.training)
        return self.fc4(x)


_model = None


def load_model():
    global _model
    model_path = os.path.join(os.path.dirname(__file__), "bytercnn_best.pth")
    if not os.path.exists(model_path):
        from huggingface_hub import hf_hub_download
        model_path = hf_hub_download(
            repo_id="Bhuvimanda/bytercnn-fft75",
            filename="bytercnn_best.pth",
        )
    _model = EfficientByteRCNN(num_classes=NUM_CLASSES)
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    if isinstance(checkpoint, dict) and "model_state" in checkpoint:
        _model.load_state_dict(checkpoint["model_state"])
    else:
        _model.load_state_dict(checkpoint)
    _model.eval()
    return _model


def get_model():
    if _model is None:
        load_model()
    return _model


def predict_block(block: bytes) -> list[dict]:
    """Predict file type for a single 512-byte block. Returns top-5 predictions."""
    model = get_model()
    block_bytes = block[:BLOCK_SIZE]
    if len(block_bytes) < BLOCK_SIZE:
        block_bytes = block_bytes + b'\x00' * (BLOCK_SIZE - len(block_bytes))

    tensor = torch.tensor(list(block_bytes), dtype=torch.long).unsqueeze(0)
    with torch.no_grad():
        probs = F.softmax(model(tensor), dim=1).squeeze(0)
    return _extract_top5(probs)


def _extract_top5(probs: torch.Tensor) -> list[dict]:
    """Extract top-5 predictions from a probability vector."""
    top5_values, top5_indices = torch.topk(probs, 5)
    results = []
    for val, cls_idx in zip(top5_values.tolist(), top5_indices.tolist()):
        label = CLASS_LABELS[cls_idx]
        results.append({
            "class_name": label,
            "category": CATEGORY_MAP[label],
            "confidence": round(val, 4)
        })
    return results


def predict_file(file_bytes: bytes, block_indices: list[int] | None = None,
                 n_samples: int = 128) -> dict:
    """Predict file type using FFT-75 compatible random-offset sampling.

    When block_indices is None (default "All Blocks" mode), samples random
    byte offsets matching how the FFT-75 dataset was constructed. When
    block_indices is provided (single/range mode), uses sequential block-
    aligned extraction for inspection purposes.
    """
    model = get_model()
    file_size = len(file_bytes)
    total_blocks = max(1, file_size // BLOCK_SIZE)

    # --- Sequential block-aligned mode (single / range) ---
    if block_indices is not None:
        indices_to_analyze = [i for i in block_indices if 0 <= i < total_blocks]
        blocks_data = []
        for idx in indices_to_analyze:
            start = idx * BLOCK_SIZE
            block = file_bytes[start:start + BLOCK_SIZE]
            if len(block) < BLOCK_SIZE:
                block = block + b'\x00' * (BLOCK_SIZE - len(block))
            blocks_data.append(list(block))

        if not blocks_data:
            return {
                "total_blocks": total_blocks,
                "analyzed_blocks": 0,
                "sampling_mode": "sequential",
                "aggregate_top5": [],
                "blocks": []
            }

        tensor = torch.tensor(blocks_data, dtype=torch.long)
        with torch.no_grad():
            all_probs = F.softmax(model(tensor), dim=1)

        block_results = []
        for i, idx in enumerate(indices_to_analyze):
            block_results.append({"index": idx, "top5": _extract_top5(all_probs[i])})

        avg_probs = all_probs.mean(dim=0)
        return {
            "total_blocks": total_blocks,
            "analyzed_blocks": len(indices_to_analyze),
            "sampling_mode": "sequential",
            "aggregate_top5": _extract_top5(avg_probs),
            "blocks": block_results
        }

    # --- FFT-75 random-offset sampling mode (default) ---
    max_offset = file_size - BLOCK_SIZE
    if max_offset < 0:
        # File smaller than one block — pad and predict once
        block = file_bytes + b'\x00' * (BLOCK_SIZE - file_size)
        tensor = torch.tensor([list(block)], dtype=torch.long)
        with torch.no_grad():
            probs = F.softmax(model(tensor), dim=1).squeeze(0)
        sig_label = detect_signature(file_bytes)
        if sig_label and sig_label in _LABEL_TO_IDX:
            sig_one_hot = torch.zeros_like(probs)
            sig_one_hot[_LABEL_TO_IDX[sig_label]] = 1.0
            probs = 0.35 * probs + 0.65 * sig_one_hot
        return {
            "total_blocks": 1,
            "analyzed_blocks": 1,
            "sampling_mode": "random",
            "signature_detected": sig_label,
            "aggregate_top5": _extract_top5(probs),
            "blocks": [{"offset": 0, "top5": _extract_top5(probs)}]
        }

    actual_samples = min(n_samples, max_offset + 1)
    offsets = sorted(random.sample(range(max_offset + 1), actual_samples))

    blocks_data = []
    for off in offsets:
        block = file_bytes[off:off + BLOCK_SIZE]
        blocks_data.append(list(block))

    tensor = torch.tensor(blocks_data, dtype=torch.long)

    with torch.no_grad():
        all_probs = F.softmax(model(tensor), dim=1)

    block_results = []
    for i, off in enumerate(offsets):
        block_results.append({"offset": off, "top5": _extract_top5(all_probs[i])})

    avg_probs = all_probs.mean(dim=0)

    # --- Combine with file signature detection ---
    sig_label = detect_signature(file_bytes)
    if sig_label and sig_label in _LABEL_TO_IDX:
        sig_idx = _LABEL_TO_IDX[sig_label]
        # Boost signature class: blend model avg with a one-hot for the
        # detected signature.  Weight chosen so header-detectable formats
        # (PNG, JPG, PDF …) surface to the top even when the model's body-
        # byte predictions disagree, while still allowing model nuance.
        sig_one_hot = torch.zeros_like(avg_probs)
        sig_one_hot[sig_idx] = 1.0
        avg_probs = 0.35 * avg_probs + 0.65 * sig_one_hot

    return {
        "total_blocks": total_blocks,
        "analyzed_blocks": actual_samples,
        "sampling_mode": "random",
        "signature_detected": sig_label,
        "aggregate_top5": _extract_top5(avg_probs),
        "blocks": block_results
    }
