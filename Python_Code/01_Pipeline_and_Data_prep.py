import os
import re
import warnings

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, find_peaks, resample, detrend

warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)

FS_EMG = 1000      
FS_XSENS = 100    
FS_XSENS_UP = 1000 

FCUT_HIGH = 40     
FCUT_LOW = 4      
FILTER_ORDER = 4   

N_STRIDES = 30     
TFL_COL = "Right Tensor fasciae latae"

EXERCISES = {
    "walking":      {"emg": "walking",       "xsens": ["walking_foot", "walkingflat_foot"]},
    "walkingincline": {"emg": "walkingincline", "xsens": ["walkingincline_foot"]},
    "stepforward":  {"emg": "stepforward",   "xsens": ["stepforward_foot"]},
    "steplateral":  {"emg": "steplateral",   "xsens": ["steplateral_foot"]},
}


FILE_OVERRIDES = {
    
    (13, "walkingincline", "xsens"): "part13_walking_foot.txt",
    # "mixed up read in xsens data": walkingincline
    # and stepforward xsens files are swapped for participant 29.
    # participant 29's own walkingincline xsens has zero peaks above the
    # 20 deg footstep-detection threshold.
    (29, "walkingincline", "xsens"): "part29_stepforward_foot.txt",
    (29, "stepforward", "xsens"): "part29_walkingincline_foot.txt",
}

EXCLUDED_TRIALS = {
    # MATLAB substituted participant 1's EMG file here (wrong person).
    # Participant 11's own file exists but has severe sensor dropout in
    # 7 of its 13 muscles (verified via detect_dropout()) not recoverable. 
    # so it is dropped rather than reported from either bad source.
    (11, "walkingincline")
}


def _find_file_anywhere(data_root: str, filename: str) -> str:
    fn_low = filename.lower()
    for root, _dirs, fns in os.walk(data_root):
        for fn in fns:
            if fn.startswith("~$"):
                continue
            if fn.lower() == fn_low:
                return os.path.join(root, fn)
    return None


# Data discovery  (create_struct.m equivalent)

def discover_participant_files(data_root: str) -> dict:
    registry = {}

    for folder in sorted(os.listdir(data_root)):
        folder_path = os.path.join(data_root, folder)
        if not os.path.isdir(folder_path):
            continue
        if "participant" not in folder.lower():
            continue

        m = re.search(r"participant(\d+)", folder, re.IGNORECASE)
        if not m:
            continue
        part_num = int(m.group(1))
        
        all_xlsx = {} 
        all_txts = {}  

        for root, dirs, fns in os.walk(folder_path):
            fns = [f for f in fns if not f.startswith("~$")]
            for fn in fns:
                fn_low = fn.lower()
                full = os.path.join(root, fn)
                if fn_low.endswith(".xlsx"):
                    all_xlsx[fn_low] = full
                elif fn_low.endswith(".txt") and "foot" in fn_low:
                    all_txts[fn_low] = full

        exercises_found = {}
        for ex_key, ex_info in EXERCISES.items():
            emg_fn = f"part{part_num}{ex_info['emg']}.xlsx"
            emg_path = all_xlsx.get(emg_fn.lower())
            if emg_path is None:
                continue

            xsens_path = None
            for xs_pattern in ex_info["xsens"]:
                xs_fn = f"part{part_num}_{xs_pattern}.txt"
                xsens_path = all_txts.get(xs_fn.lower())
                if xsens_path is not None:
                    break
            if xsens_path is None:
                continue

            exercises_found[ex_key] = {"emg": emg_path, "xsens": xsens_path}
            
        for ex_key in list(exercises_found.keys()):
            if (part_num, ex_key) in EXCLUDED_TRIALS:
                del exercises_found[ex_key]

        if exercises_found:
            registry[part_num] = {"folder": folder, **exercises_found}
            
    for (part_num, ex_key, file_type), override_fn in FILE_OVERRIDES.items():
        if part_num not in registry or ex_key not in registry[part_num]:
            continue
        override_path = _find_file_anywhere(data_root, override_fn)
        if override_path is not None:
            registry[part_num][ex_key][file_type] = override_path
        else:
            warnings.warn(
                f"Override file '{override_fn}' for participant {part_num} "
                f"{ex_key} ({file_type}) not found; using the file discovered "
                "by name-matching instead, which will NOT match the paper.",
                RuntimeWarning,
            )

    return registry


# Load raw data

def _longest_true_run(bool_arr: np.ndarray) -> int:
    if not bool_arr.any():
        return 0
    diff = np.diff(bool_arr.astype(np.int8))
    starts = np.where(diff == 1)[0] + 1
    ends = np.where(diff == -1)[0] + 1
    if bool_arr[0]:
        starts = np.r_[0, starts]
    if bool_arr[-1]:
        ends = np.r_[ends, len(bool_arr)]
    if len(starts) == 0:
        return 0
    return int((ends - starts).max())


def detect_dropout(emg: np.ndarray, zero_pct_thresh: float = 15.0, run_thresh: int = 1000) -> dict:
    n_muscles = emg.shape[0]
    details = []
    severe = []
    for i in range(n_muscles):
        row = emg[i]
        zero_mask = row == 0
        zero_pct = 100.0 * zero_mask.sum() / len(row)
        max_run = _longest_true_run(zero_mask)
        is_severe = zero_pct > zero_pct_thresh or max_run > run_thresh
        details.append({"muscle_idx": i, "zero_pct": zero_pct, "max_run": max_run, "severe": is_severe})
        if is_severe:
            severe.append(i)
    return {"n_severe": len(severe), "severe_muscles": severe, "details": details}


def load_emg(xlsx_path: str, drop_tfl: bool = True, return_names: bool = False):
    df = pd.read_excel(xlsx_path, header=10, engine="openpyxl")
    df.columns = [c.strip() for c in df.columns]
    
    muscle_cols = [c for c in df.columns if c not in ("Frame", "Time")]

    if drop_tfl and TFL_COL in muscle_cols:
        muscle_cols = [c for c in muscle_cols if c != TFL_COL]

    emg = df[muscle_cols].to_numpy(dtype=np.float64).T  
    
    corrupt_mask = np.abs(emg) > 100.0
    if corrupt_mask.any():
        n_bad = int(corrupt_mask.sum())
        warnings.warn(
            f"{os.path.basename(xlsx_path)}: {n_bad} samples with |EMG| > 100 mV "
            "detected (corrupted channel). Clamping to 0.",
            RuntimeWarning,)
        emg[corrupt_mask] = 0.0

    dropout = detect_dropout(emg)
    if dropout["n_severe"] > 0:
        bad_names = [
            [c for c in muscle_cols][i] for i in dropout["severe_muscles"]]
        warnings.warn(
            f"{os.path.basename(xlsx_path)}: severe sensor dropout detected in "
            f"{dropout['n_severe']} muscle(s) ({', '.join(bad_names)}) -- held-flat "
            "zero runs of >1s or >15% of the trial. This data cannot be recovered; "
            "VAF computed from it should be treated as unreliable.",
            RuntimeWarning,
        )

    if return_names:
        return emg, muscle_cols
    return emg


def load_xsens(txt_path: str, return_pitch: bool = False):
    df = pd.read_csv(txt_path, sep="\t", comment="/", skip_blank_lines=True)
    df.columns = [c.strip() for c in df.columns]

    roll = df["Roll"].to_numpy(dtype=np.float64)
    if return_pitch:
        if "Pitch" in df.columns:
            pitch = df["Pitch"].to_numpy(dtype=np.float64)
        else:
            pitch = np.full_like(roll, np.nan)
        return roll, pitch
    return roll


# Xsens resampling 

def resample_xsens(roll_100hz: np.ndarray) -> np.ndarray:
    
    n_out = len(roll_100hz) * (FS_XSENS_UP // FS_XSENS)
    resampled = resample(roll_100hz, n_out)
    return resampled


# EMG signal processing

def _butter_filter(signal_2d: np.ndarray, cutoff: float, btype: str) -> np.ndarray:
    b, a = butter(FILTER_ORDER, cutoff / (FS_EMG / 2), btype=btype)
    return filtfilt(b, a, signal_2d, axis=1)


def filter_emg(emg: np.ndarray) -> np.ndarray:
    # Step 1: High-pass at 40 Hz
    filt = _butter_filter(emg, FCUT_HIGH, "high")

    # Step 2: cross-muscle detrend
    filt = detrend(filt, axis=1)

    # Step 3: rectify
    filt = np.abs(filt)

    # Step 4: low-pass at 4 Hz
    filt = _butter_filter(filt, FCUT_LOW, "low")

    # Clip filter ringing artifacts
    filt = np.maximum(filt, 0.0)

    # Step 5: peak normalize per muscle
    peak_vals = filt.max(axis=1, keepdims=True)
    peak_vals[peak_vals == 0] = 1.0 
    filt = filt / peak_vals

    return filt

# stride segmentation

def segment_walking_strides(
    emg: np.ndarray,
    xsens_1000hz: np.ndarray,
    n_strides: int = N_STRIDES,
) -> np.ndarray:
    half = n_strides // 2  
    min_peaks = n_strides + 1

    n_emg = emg.shape[1]
    if len(xsens_1000hz) > n_emg:
        xsens_1000hz = xsens_1000hz[:n_emg]

    peak_idxs, _ = find_peaks(-xsens_1000hz, height=20)
    sign_used = "negative (MATLAB literal)"

    if len(peak_idxs) < min_peaks:
        peak_idxs_pos, _ = find_peaks(xsens_1000hz, height=20)
        if len(peak_idxs_pos) >= min_peaks:
            warnings.warn(
                "Foot IMU Roll signal has too few negative troughs >= 20 deg "
                f"({len(peak_idxs)} found); using positive excursions instead "
                f"({len(peak_idxs_pos)} found). This deviates from MATLAB's "
                "literal code (which hardcodes negative) but is needed for "
                "participants with reversed sensor orientation.",
                RuntimeWarning,
            )
            peak_idxs = peak_idxs_pos
            sign_used = "positive (sensor-orientation fallback)"

    n_peaks = len(peak_idxs)
    MIN_PEAKS_GRACEFUL = 11

    if n_peaks < MIN_PEAKS_GRACEFUL:
        warnings.warn(
            f"Only {n_peaks} foot-strike troughs found at MinPeakHeight=20 deg "
            f"({sign_used} sign). Too few even for a reduced stride window; "
            "using the full trial instead. MATLAB's literal code would error ",
            RuntimeWarning)
        return emg

    troughloc = peak_idxs.astype(np.int64)
    if n_peaks % 2 == 0:
        troughloc = np.append(troughloc, 0)

    median_val = np.median(troughloc)
    anchor_pos = int(np.where(troughloc == median_val)[0][0])  

    if n_peaks < min_peaks:
        effective_half = min(half, anchor_pos, n_peaks - 1 - anchor_pos)
        warnings.warn(
            f"Only {n_peaks} foot-strike troughs found at MinPeakHeight=20 deg "
            f"({sign_used} sign; need >= {min_peaks} for {n_strides} strides). "
            f"Using {effective_half * 2} strides instead of {n_strides}.",
            RuntimeWarning,
        )
    else:
        effective_half = half

    first_sample = peak_idxs[anchor_pos - effective_half]
    last_sample  = peak_idxs[anchor_pos + effective_half]

    return emg[:, first_sample:last_sample]

# Complete per-exercise processing

def process_walking(emg_path: str, xsens_path: str, return_names: bool = False):
    if return_names:
        emg_raw, muscle_cols = load_emg(emg_path, return_names=True)
    else:
        emg_raw = load_emg(emg_path)
    roll_100hz = load_xsens(xsens_path)
    roll_1000hz = resample_xsens(roll_100hz)
    emg_raw_seg = segment_walking_strides(emg_raw, roll_1000hz)  
    emg_filt = filter_emg(emg_raw_seg)                         
    if return_names:
        return emg_filt, muscle_cols
    return emg_filt


def process_step(emg_path: str, return_names: bool = False):
    if return_names:
        emg_raw, muscle_cols = load_emg(emg_path, return_names=True)
    else:
        emg_raw = load_emg(emg_path)
    emg_filt = filter_emg(emg_raw)
    if return_names:
        return emg_filt, muscle_cols
    return emg_filt


# Co-contraction Index 

KNEE_AGONIST = ["Right Vastus medialis", "Right Vastus lateralis", "Right Rectus femoris"]
KNEE_ANTAGONIST = ["Right Biceps femoris caput longus", "Right Semitendinosus"]
ANKLE_AGONIST = ["Right Tibialis anterior"]
ANKLE_ANTAGONIST = ["Right Gastrocnemius medialis", "Right Gastrocnemius lateralis", "Right Soleus"]


def compute_cci(
    emg_processed: np.ndarray,
    muscle_names: list,
    agonist_muscles: list,
    antagonist_muscles: list,
    severe_dropout_idx=None,
) -> float:
    name_to_idx = {name: i for i, name in enumerate(muscle_names)}

    ag_idx = [name_to_idx[m] for m in agonist_muscles if m in name_to_idx]
    ant_idx = [name_to_idx[m] for m in antagonist_muscles if m in name_to_idx]

    if len(ag_idx) != len(agonist_muscles) or len(ant_idx) != len(antagonist_muscles):
        return float("nan")

    if severe_dropout_idx:
        severe_set = set(severe_dropout_idx)
        if severe_set.intersection(ag_idx) or severe_set.intersection(ant_idx):
            return float("nan")

    e_ag = emg_processed[ag_idx].mean(axis=0)
    e_ant = emg_processed[ant_idx].mean(axis=0)

    denom = np.sum(e_ag + e_ant)
    if denom == 0:
        return float("nan")

    return float(2.0 * np.sum(np.minimum(e_ag, e_ant)) / denom)


# IMU kinematic features

def compute_kinematic_features(roll_100hz: np.ndarray, pitch_100hz: np.ndarray,
                                threshold: float = 20.0) -> dict:
    roll_100hz = np.asarray(roll_100hz, dtype=np.float64)
    peak_roll = float(np.max(np.abs(roll_100hz))) if len(roll_100hz) else float("nan")
    rom_roll = float(np.max(roll_100hz) - np.min(roll_100hz)) if len(roll_100hz) else float("nan")

    valid_pitch = pitch_100hz[~np.isnan(pitch_100hz)] if pitch_100hz is not None else np.array([])
    peak_pitch = float(np.max(np.abs(valid_pitch))) if len(valid_pitch) else float("nan")
    rom_pitch = float(np.max(valid_pitch) - np.min(valid_pitch)) if len(valid_pitch) else float("nan")

    peak_idxs, _ = find_peaks(-roll_100hz, height=threshold)
    stride_count = int(len(peak_idxs))
    if stride_count >= 3:
        intervals = np.diff(peak_idxs) / FS_XSENS  # seconds
        stride_timing_cv = float(np.std(intervals, ddof=1) / np.mean(intervals))
    else:
        stride_timing_cv = float("nan")

    return {
        "peak_roll": peak_roll,
        "rom_roll": rom_roll,
        "peak_pitch": peak_pitch,
        "rom_pitch": rom_pitch,
        "stride_count": stride_count,
        "stride_timing_cv": stride_timing_cv}

# NMF Muscle Synergy  (PosturalData_NMFvsPCA_GUI equivalent)

def compute_nmf_vaf(emg: np.ndarray, n_synergies: int = 1, n_replicates: int = 50) -> float:
    from sklearn.decomposition import NMF

    M = emg 
    M_T = M.T 

    best_vaf = -np.inf

    for rep in range(n_replicates):
        try:
            model = NMF(
                n_components=n_synergies,
                init="random",
                max_iter=1000,
                tol=1e-6,
                random_state=rep * 7 + 13)
            W = model.fit_transform(M_T)   
            H = model.components_          
            M_recon = (W @ H).T         

            inner   = np.sum(M * M_recon)
            denom   = np.sum(M ** 2) * np.sum(M_recon ** 2)

            if denom == 0:
                vaf = 0.0
            else:
                vaf = (inner ** 2 / denom) * 100.0

            if vaf > best_vaf:
                best_vaf = vaf
        except Exception:
            continue

    return float(best_vaf)

# 8. DMCI calculation  (calc_dmci.m equivalent)

def compute_dmci(vaf_individual: float, vaf_young_group: np.ndarray) -> float:
    avg = float(np.mean(vaf_young_group))
    sd = float(np.std(vaf_young_group, ddof=1))  
    if sd == 0:
        return np.nan
    return 100.0 + 10.0 * (avg - vaf_individual) / sd


import os
import sys
import time
import traceback

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))

from emg_pipeline import (
    discover_participant_files,
    process_walking,
    process_step,
    compute_nmf_vaf,
    compute_dmci,
    compute_cci,
    compute_kinematic_features,
    load_emg,
    load_xsens,
    detect_dropout,
    EXCLUDED_TRIALS,
    KNEE_AGONIST,
    KNEE_ANTAGONIST,
    ANKLE_AGONIST,
    ANKLE_ANTAGONIST)

EXTRA_COLS = [
    "cci_knee", "cci_ankle",
    "peak_roll", "rom_roll", "peak_pitch", "rom_pitch",
    "stride_count", "stride_timing_cv"]

# Configuration
DATA_ROOT = r"c:\Users\Deepak Babu P K\OneDrive\Desktop\Laura_Handover\DATACOLLECTION"
PART_INFO = r"c:\Users\Deepak Babu P K\OneDrive\Desktop\Laura_Handover\DATACOLLECTION\ParticipantInfo.xlsx"
PART_INFO_FALLBACK = r"c:\Users\Deepak Babu P K\OneDrive\Desktop\Laura_Handover\PythonAnalysis\participants_clean.csv"
OUT_DIR   = r"c:\Users\Deepak Babu P K\OneDrive\Desktop\Laura_Handover\PythonAnalysis\pipeline_outputs"

N_REPLICATES = 50   
N_SYNERGIES  = 1    

# Participants explicitly discarded in MATLAB create_struct.m
DISCARDED_PARTS = {1, 23, 28, 34}

os.makedirs(OUT_DIR, exist_ok=True)

# Load participant info (age and group)

def load_participant_info(xlsx_path: str) -> pd.DataFrame:
    try:
        df = pd.read_excel(xlsx_path, engine="openpyxl")
        df.columns = [str(c).strip() for c in df.columns]
        col_map = {}
        for col in df.columns:
            cl = col.lower()
            if "participant" in cl or col.lower() in ("part", "id", "subject"):
                col_map.setdefault("part_num", col)
            elif "age" in cl:
                col_map.setdefault("age", col)
            # "YA = 1 OA=2" column: young adult=1, older adult=2
            elif "group" in cl or ("ya" in cl and "oa" in cl):
                col_map.setdefault("group", col)

        if "part_num" in col_map and "age" in col_map:
            out = df[[col_map["part_num"], col_map["age"]]].copy()
            out.columns = ["part_num", "age"]
            if "group" in col_map:
                out["group"] = df[col_map["group"]].values
            # Extract numeric participant number
            out["part_num"] = out["part_num"].astype(str).str.extract(r"(\d+)").astype(float)
            out = out.dropna(subset=["part_num"])
            out["part_num"] = out["part_num"].astype(int)
            out = out.set_index("part_num")
            
            try:
                fb = pd.read_csv(PART_INFO_FALLBACK)
                fb["part_num"] = fb["participant"].astype(str).str.extract(r"(\d+)").astype(int)
                fb = fb.set_index("part_num")[["age", "group"]]
                for col in ("age", "group"):
                    if col not in out.columns:
                        out[col] = np.nan
                    missing = out[col].isna()
                    out.loc[missing, col] = fb.reindex(out.index)[col][missing]
            except Exception as e:
                print(f"  Warning: could not load fallback participant info: {e}")

            out = out.dropna(subset=["age"])
            return out
    except Exception as e:
        print(f"  Warning: could not load ParticipantInfo.xlsx: {e}")
    return pd.DataFrame(columns=["age"])

# Main pipeline

def run_all(test_mode: bool = False):
    
    print("=" * 60)
    print("EMG Pipeline: MATLAB -> Python conversion")
    print("=" * 60)

    # Step 1: discover files
    registry = discover_participant_files(DATA_ROOT)
    all_parts = sorted(registry.keys())
    
    # Remove explicitly discarded participants
    
    valid_parts = [p for p in all_parts if p not in DISCARDED_PARTS]
    print(f"Found {len(all_parts)} participant folders, "
          f"{len(valid_parts)} valid (after removing discarded).")

    if test_mode:
        valid_parts = [p for p in valid_parts if p == 2]
        print(f"TEST MODE: processing participant(s): {valid_parts}")

    # Step 2: load participant ages
    part_info = load_participant_info(PART_INFO)
    print(f"Loaded info for {len(part_info)} participants.")

    # Step 3: compute VAF for each participant-exercise
    print(f"NMF replicates: {N_REPLICATES}")

    exercise_keys = ["walking", "walkingincline", "stepforward", "steplateral"]
    results = []

    for part_num in valid_parts:
        part_data = registry[part_num]
        row = {"participant": part_num}

        # Get age from ParticipantInfo if available
        if part_num in part_info.index:
            row["age"] = part_info.loc[part_num, "age"]
            if "group" in part_info.columns:
                row["group"] = part_info.loc[part_num, "group"]
        else:
            row["age"] = np.nan
            row["group"] = np.nan

        t0 = time.time()
        print(f"\n  Participant {part_num:3d}  (folder: {part_data['folder']})")

        for ex_key in exercise_keys:
            if ex_key not in part_data:
                if (part_num, ex_key) in EXCLUDED_TRIALS:
                    print(f"{ex_key:20s}: EXCLUDED, no reliable data source")
                    row[f"quality_{ex_key}"] = "excluded_no_reliable_data"
                else:
                    print(f"{ex_key:20s}: MISSING files, skipping")
                row[f"vaf_{ex_key}"] = np.nan
                for col in EXTRA_COLS:
                    row[f"{col}_{ex_key}"] = np.nan
                continue

            ex_files = part_data[ex_key]
            try:
                # Check for sensor dropout in the raw EMG 
                raw_emg = load_emg(ex_files["emg"])
                dropout = detect_dropout(raw_emg)
                if dropout["n_severe"] > 0:
                    row[f"quality_{ex_key}"] = f"severe_dropout({dropout['n_severe']}_muscles)"
                else:
                    row[f"quality_{ex_key}"] = "ok"

                if ex_key in ("walking", "walkingincline"):
                    emg_processed, muscle_cols = process_walking(
                        ex_files["emg"], ex_files["xsens"], return_names=True)
                else:
                    emg_processed, muscle_cols = process_step(
                        ex_files["emg"], return_names=True)

                vaf = compute_nmf_vaf(emg_processed, n_synergies=N_SYNERGIES,
                                      n_replicates=N_REPLICATES)
                row[f"vaf_{ex_key}"] = round(vaf, 4)

                # Co-contraction indices
                cci_knee = compute_cci(emg_processed, muscle_cols,
                                       KNEE_AGONIST, KNEE_ANTAGONIST,
                                       severe_dropout_idx=dropout["severe_muscles"])
                cci_ankle = compute_cci(emg_processed, muscle_cols,
                                        ANKLE_AGONIST, ANKLE_ANTAGONIST,
                                        severe_dropout_idx=dropout["severe_muscles"])
                row[f"cci_knee_{ex_key}"] = round(cci_knee, 4) if not np.isnan(cci_knee) else np.nan
                row[f"cci_ankle_{ex_key}"] = round(cci_ankle, 4) if not np.isnan(cci_ankle) else np.nan

                # IMU kinematic features from the foot sensor 
                roll_100hz, pitch_100hz = load_xsens(ex_files["xsens"], return_pitch=True)
                kin = compute_kinematic_features(roll_100hz, pitch_100hz)
                for k, v in kin.items():
                    row[f"{k}_{ex_key}"] = v

                flag = "  *** SEVERE DROPOUT, UNRELIABLE ***" if dropout["n_severe"] > 0 else ""
                print(f"    {ex_key:20s}: VAF = {vaf:.2f}%  CCI_knee = {cci_knee:.3f}  "
                      f"CCI_ankle = {cci_ankle:.3f}  "
                      f"[{emg_processed.shape[0]} muscles x {emg_processed.shape[1]} samples]{flag}")

            except Exception as e:
                print(f"    {ex_key:20s}: ERROR - {e}")
                if "--debug" in sys.argv:
                    traceback.print_exc()
                row[f"vaf_{ex_key}"] = np.nan
                row[f"quality_{ex_key}"] = "error"
                for col in EXTRA_COLS:
                    row[f"{col}_{ex_key}"] = np.nan

        elapsed = time.time() - t0
        print(f"    Done in {elapsed:.1f}s")
        results.append(row)

    # compute DMCI using young group as control
    df = pd.DataFrame(results)
    df = df.set_index("participant")

    # Identify young participants (group=1 or age <= 40)
    if "group" in df.columns and df["group"].notna().any():
        young_mask = df["group"] == 1
    elif "age" in df.columns:
        young_mask = df["age"] <= 40
    else:
        print("WARNING: cannot identify young group; using all participants as control")
        young_mask = pd.Series(True, index=df.index)

    young_parts = df.index[young_mask].tolist()
    print(f"Young control group: {len(young_parts)} participants: {young_parts}")

    for ex_key in exercise_keys:
        vaf_col = f"vaf_{ex_key}"
        dmci_col = f"dmci_{ex_key}"
        quality_col = f"quality_{ex_key}"

        if vaf_col not in df.columns:
            df[dmci_col] = np.nan
            continue

        # Exclude trials with severe sensor dropout from the young-group
        if quality_col in df.columns:
            reliable_mask = df[quality_col].apply(lambda q: not str(q).startswith("severe_dropout"))
        else:
            reliable_mask = pd.Series(True, index=df.index)

        baseline_mask = young_mask & reliable_mask
        excluded = df.index[young_mask & ~reliable_mask].tolist()
        if excluded:
            print(f"{ex_key}: excluding {excluded} from young-group DMCI baseline "
                  "(severe sensor dropout)")

        young_vafs = df.loc[baseline_mask, vaf_col].dropna().values
        if len(young_vafs) < 2:
            print(f"WARNING: fewer than 2 reliable young participants with VAF for {ex_key}; "
                  "DMCI will be NaN.")
            df[dmci_col] = np.nan
            continue

        dmci_vals = [
            compute_dmci(vaf, young_vafs) if not np.isnan(vaf) else np.nan
            for vaf in df[vaf_col]]
        df[dmci_col] = dmci_vals
        
    ordered_cols = ["age"]
    if "group" in df.columns:
        ordered_cols.append("group")
    for ex in exercise_keys:
        ordered_cols += [f"vaf_{ex}", f"dmci_{ex}", f"quality_{ex}"]
        ordered_cols += [f"{col}_{ex}" for col in EXTRA_COLS]
    df = df[[c for c in ordered_cols if c in df.columns]]
    df = df.reset_index()

    # Save
    out_csv = os.path.join(OUT_DIR, "pipeline_results.csv")
    try:
        df.to_csv(out_csv, index=False)
        print(f"\nSaved results to: {out_csv}")
    except PermissionError:
        fallback_csv = os.path.join(OUT_DIR, f"pipeline_results{int(time.time())}.csv")
        df.to_csv(fallback_csv, index=False)
        print(f"\n{out_csv} was locked (open elsewhere?); saved to: {fallback_csv}")
    print("\nResults summary:")
    print(df.to_string(index=False, float_format=lambda x: f"{x:.2f}"))

    return df

# Testing (single participant)

def test_participant(part_num: int = 3):

    registry = discover_participant_files(DATA_ROOT)
    if part_num not in registry:
        print(f"Participant {part_num} not found in registry.")
        return

    part_data = registry[part_num]
    print(f"Folder: {part_data['folder']}")

    for ex_key in ["walking", "walkingincline", "stepforward", "steplateral"]:
        if ex_key not in part_data:
            print(f"  {ex_key}: missing")
            continue

        files = part_data[ex_key]
        print(f"\n  Exercise: {ex_key}")
        print(f"  EMG:   {os.path.basename(files['emg'])}")
        print(f"  Xsens: {os.path.basename(files['xsens'])}")

        try:
            if ex_key in ("walking", "walkingincline"):
                emg = process_walking(files["emg"], files["xsens"])
            else:
                emg = process_step(files["emg"])

            vaf = compute_nmf_vaf(emg, n_synergies=1, n_replicates=5)
            print(f"  EMG shape: {emg.shape}  (muscles x samples)")
            print(f"  VAF (1 synergy, 5 replicates): {vaf:.2f}%")
            print(f"  EMG range: [{emg.min():.4f}, {emg.max():.4f}]")

        except Exception as e:
            print(f"  ERROR: {e}")
            traceback.print_exc()


import pandas as pd
import numpy as np

RAW_PATH = r"c:\Users\Deepak Babu P K\OneDrive\Desktop\Laura_Handover\PythonAnalysis\pipeline_outputs\pipeline_results.csv"
OUT_PATH = r"c:\Users\Deepak Babu P K\OneDrive\Desktop\Laura_Handover\PythonAnalysis\participants_clean.csv"

EXERCISES = ["walking", "walkingincline", "stepforward", "steplateral"]

# Load data 
df = pd.read_csv(RAW_PATH)
df.columns = [str(c).strip() for c in df.columns]
vaf_cols  = [f"vaf_{ex}"  for ex in EXERCISES]
dmci_cols = [f"dmci_{ex}" for ex in EXERCISES]
quality_cols = [f"quality_{ex}" for ex in EXERCISES]

core_cols = ["participant", "age", "group"] + vaf_cols + dmci_cols + quality_cols
missing_core = [c for c in core_cols if c not in df.columns]
if missing_core:
    raise ValueError(f"Expected columns missing from {RAW_PATH}: {missing_core}")
df = df[core_cols].copy()

# Keep only rows with a valid age 
df["age"]   = pd.to_numeric(df["age"], errors="coerce")
df["group"] = pd.to_numeric(df["group"], errors="coerce")
df = df[df["age"].notna()].copy()

print(f"Participants loaded: {len(df)}")
print(f"  Young (group=1): {(df['group']==1).sum()}")
print(f"  Older (group=2): {(df['group']==2).sum()}")
print(f"  Age range: {df['age'].min():.0f} - {df['age'].max():.0f} yrs")

# Feature engineering 
df["vaf_ratio_sfwd_walk"]  = df["vaf_stepforward"]  / df["vaf_walking"]
df["dmci_ratio_sfwd_walk"] = df["dmci_stepforward"] / df["dmci_walking"]
df["vaf_ratio_sfwd_slat"]  = df["vaf_stepforward"]  / df["vaf_steplateral"]
df["dmci_ratio_sfwd_slat"] = df["dmci_stepforward"] / df["dmci_steplateral"]

# Mean and range of VAF/DMCI across all exercises (where available)
df["vaf_mean_all"]  = df[vaf_cols].mean(axis=1)
df["dmci_mean_all"] = df[dmci_cols].mean(axis=1)
df["vaf_range"]  = df[vaf_cols].max(axis=1)  - df[vaf_cols].min(axis=1)
df["dmci_range"] = df[dmci_cols].max(axis=1) - df[dmci_cols].min(axis=1)

# Report missingness
print("\nMissing values per feature:")
print(df.isnull().sum()[df.isnull().sum() > 0])

print("\nQuality flags (non-'ok' trials):")
for qc in quality_cols:
    bad = df[df[qc].notna() & (df[qc] != "ok")]
    if len(bad):
        print(f"  {qc}: {bad[['participant', qc]].to_dict('records')}")

# Save 
df.to_csv(OUT_PATH, index=False)
print(f"\nSaved -> {OUT_PATH}")

print(f"\nShape: {df.shape}")
print("\nFirst 5 rows:")
print(df.head(5).to_string())