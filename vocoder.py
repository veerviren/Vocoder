import os
import numpy as np
import pandas as pd
import librosa
import librosa.display
from scipy.io.wavfile import read
from scipy.spatial.distance import euclidean
import soundfile as sf
import matplotlib.pyplot as plt


class AudioProcessingPipeline:
    def __init__(self, original_folder, output_folder, n_mels=500):
        self.original_folder = original_folder
        self.output_folder = output_folder
        self.n_mels = n_mels

        # Define subfolders
        self.npy_folder = os.path.join(output_folder, 'npy', str(n_mels))
        self.wav_folder = os.path.join(output_folder, 'wav', str(n_mels))
        self.png_folder = os.path.join(output_folder, 'png', str(n_mels))
        self.results = []

        # Create folders if they don't exist
        os.makedirs(self.npy_folder, exist_ok=True)
        os.makedirs(self.wav_folder, exist_ok=True)
        os.makedirs(self.png_folder, exist_ok=True)

    def compute_mel_spectrogram(self, file_path, sr=22050):
        """Compute Mel-spectrogram for an audio file."""
        y, sr = librosa.load(file_path, sr=sr)
        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=self.n_mels)
        mel_db = librosa.power_to_db(mel_spec, ref=np.max)
        return mel_db, sr

    def griffin_lim_reconstruction(self, mel_spec, sr, n_iter=32):
        """Reconstruct audio from Mel-spectrogram using Griffin-Lim."""
        mel_power = librosa.db_to_power(mel_spec)
        linear_spec = librosa.feature.inverse.mel_to_stft(mel_power, sr=sr)
        reconstructed_audio = librosa.griffinlim(linear_spec, n_iter=n_iter)
        return reconstructed_audio

    def normalize(self, data):
        """Normalize data to resemble a probability distribution."""
        data = np.abs(data)
        return data / np.sum(data)

    def kl_divergence(self, p, q):
        """Calculate KL Divergence."""
        p = np.clip(p, 1e-10, None)
        q = np.clip(q, 1e-10, None)
        return np.sum(p * np.log(p / q))

    def process_audio_files(self):
        """Process all audio files to compute metrics and save outputs."""
        original_files = sorted(os.listdir(self.original_folder))
        generated_files = sorted(os.listdir(self.wav_folder))

        for file in original_files:
            if file.endswith(".wav"):
                file_path = os.path.join(self.original_folder, file)
                mel_spec, sr = self.compute_mel_spectrogram(file_path)

                # Save Mel-spectrogram as .npy
                npy_path = os.path.join(self.npy_folder, f"{os.path.splitext(file)[0]}_mel.npy")
                np.save(npy_path, mel_spec)

                # Save Mel-spectrogram as .png
                self.save_mel_plot(mel_spec, sr, file)

                # Reconstruct audio and save as .wav
                reconstructed_audio = self.griffin_lim_reconstruction(mel_spec, sr)
                wav_path = os.path.join(self.wav_folder, f"{os.path.splitext(file)[0]}.wav")
                sf.write(wav_path, reconstructed_audio, sr)

    def save_mel_plot(self, mel_spec, sr, file_name):
        """Save Mel-spectrogram plot as PNG."""
        plt.figure(figsize=(10, 4))
        librosa.display.specshow(mel_spec, sr=sr, x_axis='time', y_axis='mel', cmap='coolwarm')
        plt.colorbar(format='%+2.0f dB')
        plt.title(f"Mel-spectrogram of {file_name}")
        plt.tight_layout()
        plt.savefig(os.path.join(self.png_folder, f"{os.path.splitext(file_name)[0]}_mel.png"))
        plt.close()

    def compare_audio_files(self):
        """Compare original and generated audio files for both audio and Mel-spectrogram."""
        original_files = sorted(os.listdir(self.original_folder))
        generated_files = sorted(os.listdir(self.wav_folder))

        audio_results = []
        mel_results = []

        for orig_file in original_files:
            if not orig_file.endswith(".wav"):
                continue
            
            orig_number = orig_file.split('.')[0]
            matching_files = [gen_file for gen_file in generated_files if gen_file.startswith(orig_number)]
            if not matching_files:
                print(f"No matching file found for {orig_file}")
                continue

            gen_file = matching_files[0]
            orig_path = os.path.join(self.original_folder, orig_file)
            gen_path = os.path.join(self.wav_folder, gen_file)

            # Load original and generated audio
            orig_audio, orig_sr = librosa.load(orig_path, sr=None)
            gen_audio, gen_sr = librosa.load(gen_path, sr=None)

            # Resample if sampling rates do not match
            target_sr = 22050
            if orig_sr != target_sr:
                orig_audio = librosa.resample(orig_audio, orig_sr=orig_sr, target_sr=target_sr)
                orig_sr = target_sr
            if gen_sr != target_sr:
                gen_audio = librosa.resample(gen_audio, orig_sr=gen_sr, target_sr=target_sr)
                gen_sr = target_sr

            # Ensure dimensions match
            min_samples = min(len(orig_audio), len(gen_audio))
            orig_audio = orig_audio[:min_samples]
            gen_audio = gen_audio[:min_samples]

            # Compute metrics for audio
            audio_euc_dist = euclidean(orig_audio, gen_audio)
            orig_audio_norm = self.normalize(orig_audio)
            gen_audio_norm = self.normalize(gen_audio)
            audio_kl_div = self.kl_divergence(orig_audio_norm, gen_audio_norm)

            audio_results.append({
                "Original File": orig_file.split('.')[0],
                "Generated File": gen_file.split('.')[0],
                "Euclidean Distance": audio_euc_dist,
                "KL Divergence": audio_kl_div
            })

            # Compute Mel-spectrograms
            original_mel, _ = self.compute_mel_spectrogram(orig_path, sr=target_sr)
            generated_mel, _ = self.compute_mel_spectrogram(gen_path, sr=target_sr)

            # Ensure dimensions match
            min_time_frames = min(original_mel.shape[1], generated_mel.shape[1])
            original_mel = original_mel[:, :min_time_frames]
            generated_mel = generated_mel[:, :min_time_frames]

            # Compute metrics for Mel-spectrogram
            mel_euc_dist = euclidean(original_mel.flatten(), generated_mel.flatten())
            orig_mel_norm = self.normalize(original_mel.flatten())
            gen_mel_norm = self.normalize(generated_mel.flatten())
            mel_kl_div = self.kl_divergence(orig_mel_norm, gen_mel_norm)

            mel_results.append({
                "Original File": orig_file.split('.')[0],
                "Generated File": gen_file.split('.')[0],
                "Euclidean Distance": mel_euc_dist,
                "KL Divergence": mel_kl_div
            })

        # Store results
        self.audio_results = audio_results
        self.mel_results = mel_results

    def save_results(self):
        """Save comparison results to two separate CSV files."""
        audio_df = pd.DataFrame(self.audio_results)
        audio_csv = os.path.join(self.output_folder, f"audio_comparison_results_{self.n_mels}.csv")
        audio_df.to_csv(audio_csv, index=False)
        print(f"Audio comparison results saved to {audio_csv}")

        mel_df = pd.DataFrame(self.mel_results)
        mel_csv = os.path.join(self.output_folder, f"mel_comparison_results_{self.n_mels}.csv")
        mel_df.to_csv(mel_csv, index=False)
        print(f"Mel-spectrogram comparison results saved to {mel_csv}")


if __name__ == "__main__":
    original_folder = os.path.join('../data') # Change this to the path of the folder containing the original audio files
    output_folder = os.path.join('./output')

    vocoder = AudioProcessingPipeline(original_folder, output_folder, n_mels=500)
    vocoder.process_audio_files()
    vocoder.compare_audio_files()
    vocoder.save_results()
