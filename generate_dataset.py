import os
import numpy as np
import soundfile as sf
from scipy.signal import butter, lfilter

FS = 16000

def create_formant_filter(f0_center, bw, fs=FS):
    # Resonator filter
    r = np.exp(-np.pi * bw / fs)
    theta = 2 * np.pi * f0_center / fs
    b = [1.0 - r]
    a = [1.0, -2.0 * r * np.cos(theta), r * r]
    return b, a

def generate_utterance(word, rep_idx, fs=FS):
    rng = np.random.RandomState(42 + hash(word) % 1000 + rep_idx * 17)
    
    # Timing variations (tempo)
    lead_silence = 0.25 + rng.uniform(-0.05, 0.05)
    tail_silence = 0.25 + rng.uniform(-0.05, 0.05)
    base_duration = {'khong': 0.55, 'mot': 0.42, 'hai': 0.50, 'ba': 0.45, 'bon': 0.52}[word]
    duration = base_duration * (1.0 + rng.uniform(-0.08, 0.12))
    
    t_speech = np.linspace(0, duration, int(duration * fs), endpoint=False)
    n_samples = len(t_speech)
    
    # Base Pitch F0 (Hz)
    f0_base = 135.0 + rng.uniform(-6, 6)
    
    # Phoneme specifications for each Vietnamese digit
    if word == 'khong':
        # 'kh' (unvoiced fricative ~80ms) + 'ong' (voiced vowel /o/ + nasal /ŋ/)
        fric_len = int(0.08 * fs)
        fric_noise = rng.normal(0, 0.08, fric_len)
        b_hp, a_hp = butter(3, [1200 / (fs/2), 4000 / (fs/2)], btype='band')
        fric_sound = lfilter(b_hp, a_hp, fric_noise) * np.linspace(0.2, 0.8, fric_len)
        
        v_len = n_samples - fric_len
        t_v = np.linspace(0, v_len / fs, v_len, endpoint=False)
        pitch_contour = f0_base * (1.0 - 0.25 * (t_v / (v_len / fs))**1.5) # Falling tone (thanh huyền/ngang)
        phase = 2 * np.pi * np.cumsum(pitch_contour) / fs
        # Glottal source
        source = (np.sin(phase) + 0.5 * np.sin(2*phase) + 0.3 * np.sin(3*phase) + 0.15 * np.sin(4*phase))
        
        # Formants for /oŋ/
        b1, a1 = create_formant_filter(500 + rng.uniform(-20, 20), 80, fs)
        b2, a2 = create_formant_filter(950 + rng.uniform(-30, 30), 100, fs)
        b3, a3 = create_formant_filter(2400, 150, fs)
        v_sound = lfilter(b1, a1, source) + 0.6 * lfilter(b2, a2, source) + 0.2 * lfilter(b3, a3, source)
        
        # Envelope
        env = np.sin(np.pi * (t_v / (v_len / fs))) ** 0.6
        v_sound = v_sound * env
        speech = np.concatenate([fric_sound, v_sound])
        
    elif word == 'mot':
        # 'm' (nasal murmur ~70ms) + 'o' (/ɔ/) + 't' (sharp unvoiced stop coda ~50ms)
        m_len = int(0.07 * fs)
        t_m = np.linspace(0, m_len / fs, m_len, endpoint=False)
        p_m = 2 * np.pi * np.cumsum(np.full_like(t_m, f0_base * 0.95)) / fs
        s_m = 0.4 * np.sin(p_m) + 0.2 * np.sin(2 * p_m)
        b_m, a_m = create_formant_filter(280, 60, fs)
        m_sound = lfilter(b_m, a_m, s_m) * np.linspace(0.2, 0.7, m_len)
        
        v_len = n_samples - m_len - int(0.05 * fs)
        t_v = np.linspace(0, v_len / fs, v_len, endpoint=False)
        pitch_contour = f0_base * (1.1 - 0.4 * (t_v / (v_len / fs))**2) # heavy tone (thanh nặng: drops fast)
        phase = 2 * np.pi * np.cumsum(pitch_contour) / fs
        source = (np.sin(phase) + 0.5 * np.sin(2*phase) + 0.25 * np.sin(3*phase))
        b1, a1 = create_formant_filter(520, 90, fs)
        b2, a2 = create_formant_filter(1050, 110, fs)
        b3, a3 = create_formant_filter(2500, 160, fs)
        v_sound = lfilter(b1, a1, source) + 0.5 * lfilter(b2, a2, source) + 0.15 * lfilter(b3, a3, source)
        env = np.linspace(0.8, 0.1, v_len)
        v_sound = v_sound * env
        
        # 't' stop burst
        t_len = n_samples - m_len - v_len
        t_burst = rng.normal(0, 0.06, t_len)
        b_t, a_t = butter(2, 2500 / (fs/2), btype='high')
        t_sound = lfilter(b_t, a_t, t_burst) * np.exp(-np.linspace(0, 5, t_len))
        speech = np.concatenate([m_sound, v_sound, t_sound])
        
    elif word == 'hai':
        # 'h' (unvoiced aspirate ~60ms) + diphthong 'ai' (F1: 750->350, F2: 1250->2200)
        h_len = int(0.06 * fs)
        h_noise = rng.normal(0, 0.06, h_len)
        b_h, a_h = butter(2, [1000 / (fs/2), 3500 / (fs/2)], btype='band')
        h_sound = lfilter(b_h, a_h, h_noise) * np.linspace(0.2, 0.7, h_len)
        
        v_len = n_samples - h_len
        t_v = np.linspace(0, v_len / fs, v_len, endpoint=False)
        pitch_contour = f0_base * (1.0 + 0.05 * np.sin(np.pi * t_v / (v_len / fs))) # ngang
        phase = 2 * np.pi * np.cumsum(pitch_contour) / fs
        source = (np.sin(phase) + 0.6 * np.sin(2*phase) + 0.3 * np.sin(3*phase) + 0.2 * np.sin(4*phase))
        
        # Moving formants for /ai/
        b1_start, a1_start = create_formant_filter(780, 100, fs)
        b1_end, a1_end = create_formant_filter(360, 80, fs)
        b2_start, a2_start = create_formant_filter(1250, 120, fs)
        b2_end, a2_end = create_formant_filter(2200, 140, fs)
        
        s1 = 0.5 * (lfilter(b1_start, a1_start, source) * (1 - t_v / (v_len / fs)) + lfilter(b1_end, a1_end, source) * (t_v / (v_len / fs)))
        s2 = 0.4 * (lfilter(b2_start, a2_start, source) * (1 - t_v / (v_len / fs)) + lfilter(b2_end, a2_end, source) * (t_v / (v_len / fs)))
        v_sound = s1 + s2
        env = np.sin(np.pi * (t_v / (v_len / fs))) ** 0.5
        speech = np.concatenate([h_sound, v_sound * env])
        
    elif word == 'ba':
        # 'b' voiced plosive burst (~30ms) + pure vowel 'a' (F1 ~ 800, F2 ~ 1300)
        b_len = int(0.03 * fs)
        t_b = np.linspace(0, b_len / fs, b_len, endpoint=False)
        b_sound = 0.4 * np.sin(2 * np.pi * 120 * t_b) + rng.normal(0, 0.03, b_len)
        
        v_len = n_samples - b_len
        t_v = np.linspace(0, v_len / fs, v_len, endpoint=False)
        pitch_contour = np.full_like(t_v, f0_base)
        phase = 2 * np.pi * np.cumsum(pitch_contour) / fs
        source = (np.sin(phase) + 0.5 * np.sin(2*phase) + 0.3 * np.sin(3*phase) + 0.15 * np.sin(4*phase))
        b1, a1 = create_formant_filter(820, 90, fs)
        b2, a2 = create_formant_filter(1320, 110, fs)
        b3, a3 = create_formant_filter(2600, 150, fs)
        v_sound = lfilter(b1, a1, source) + 0.5 * lfilter(b2, a2, source) + 0.2 * lfilter(b3, a3, source)
        env = np.sin(np.pi * (t_v / (v_len / fs))) ** 0.55
        speech = np.concatenate([b_sound, v_sound * env])
        
    else: # 'bon'
        # 'b' plosive + vowel 'ɔ' + high falling/rising tone (thanh sắc) + 'n' nasal coda
        b_len = int(0.03 * fs)
        t_b = np.linspace(0, b_len / fs, b_len, endpoint=False)
        b_sound = 0.4 * np.sin(2 * np.pi * 120 * t_b)
        
        v_len = n_samples - b_len
        t_v = np.linspace(0, v_len / fs, v_len, endpoint=False)
        pitch_contour = f0_base * (0.95 + 0.35 * (t_v / (v_len / fs))**1.2) # sắc (rising tone)
        phase = 2 * np.pi * np.cumsum(pitch_contour) / fs
        source = (np.sin(phase) + 0.5 * np.sin(2*phase) + 0.25 * np.sin(3*phase))
        b1, a1 = create_formant_filter(540, 85, fs)
        b2, a2 = create_formant_filter(980, 100, fs)
        b3, a3 = create_formant_filter(2500, 150, fs)
        v_sound = lfilter(b1, a1, source) + 0.5 * lfilter(b2, a2, source) + 0.2 * lfilter(b3, a3, source)
        env = np.sin(np.pi * (t_v / (v_len / fs))) ** 0.7
        speech = np.concatenate([b_sound, v_sound * env])
        
    # Scale speech
    speech = speech / (np.max(np.abs(speech)) + 1e-9) * 0.75
    
    # Silence with realistic faint ambient background noise
    n_lead = int(lead_silence * fs)
    n_tail = int(tail_silence * fs)
    lead = rng.normal(0, 0.003, n_lead)
    tail = rng.normal(0, 0.003, n_tail)
    
    total_audio = np.concatenate([lead, speech, tail])
    return total_audio.astype(np.float32)

def main():
    words = ['khong', 'mot', 'hai', 'ba', 'bon']
    for word in words:
        out_dir = os.path.join('dataset', word)
        os.makedirs(out_dir, exist_ok=True)
        for i in range(1, 6):
            fname = f"{word}_0{i}.wav"
            fpath = os.path.join(out_dir, fname)
            audio = generate_utterance(word, i)
            sf.write(fpath, audio, FS)
            print(f"Generated {fpath} (len: {len(audio)/FS:.2f}s)")

if __name__ == '__main__':
    main()
