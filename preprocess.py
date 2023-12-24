import os
import numpy as np

from modules.utils import process_audio_data

if __name__ == "__main__":

    if not os.path.exists("data/normal"):
        os.makedirs("data/normal")
    
    if not os.path.exists("data/whistle"):
        os.makedirs("data/whistle")

    normal_count = 1
    for idx, sound_file in enumerate(os.listdir("sound_data/1000")):
        if idx >= 20:
            break
        spec_db = process_audio_data(os.path.join("sound_data/1000", sound_file))
        for i in range(0, spec_db.shape[1] - 200, 200):
            spec_db_frame = spec_db[:, i:i+200]
            np.save(os.path.join("data/normal", f"{normal_count}.npy"), spec_db_frame.T)
            normal_count += 1
    
    whistle_count = 1
    for sound_file in os.listdir("sound_data/whistle"):
        spec_db = process_audio_data(os.path.join("sound_data/whistle", sound_file))
        for i in range(0, spec_db.shape[1] - 200, 50):
            spec_db_frame = spec_db[:, i:i+200]
            np.save(os.path.join("data/whistle", f"{whistle_count}.npy"), spec_db_frame.T)
            whistle_count += 1
        