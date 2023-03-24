# AudioDataCollation

## Step 1: downloading audio files

Download from youtube:
- [youtube-dl](https://github.com/ytdl-org/youtube-dl) 
- [yt-dlp](https://github.com/yt-dlp/yt-dlp)


## Step 2: audio segmentation

### Step 2-1: using whisper to detect the audio segments 
```
whisper_asr.py --wav_dir /path/to/your/wav/file/dir
```
This will save the results in {dir}_whisper.csv

### Step 2-2: using the whisper to do the wavfile segmentation

```
cut_wav.py --wav_dir /path/to/your/wav/file/dir --input_file /path/to/your/{dir}_whisper.csv
```

This will save the wav segments in /path/to/your/wav/file/{dir}_seg


## Step 3: speech enhancement

Remove the background noise of the audio. 
```
python CMGAN/enhanced_speech_cpu.py --test_dir /path/to/wavfiles/dir #if you use cpu
python CMGAN/enhanced_speech_gpu.py --test_dir /path/to/wavfiles/dir #if you use gpu
```

The enhanced signals will be saved in the ./data/{dir}_en directory.

See [CMGAN](https://github.com/ruizhecao96/CMGAN) for more details. 

## Step 4: SNR level detection

Although speech enhancement can remove background noise from a noisy signal, it also cause distoration. 
Such distortion is especially bad for a high Signal-to-noise ratio (SNR) signal because these signals don't have the background noise and thus speech enhancement will only cause negative impact. One possible method to address this issue is by using a SNR level detection model. 
The following model takes speech signals and corresponsing enhanced speech signals as input. Then, output the SNR level score, which is in an range of 0-1, higher score indicate better quality.  
Note that this SNR level detection model was designed base on the idea that "the original and enhanced signals of high SNR signals are similar because high SNR signals don't have much background noise to be remove." Even though overall it's true, but it also exists condition that the original and enhanced signals are similar because the speech enhancement model fails to revmoe the background noise. 


See [NRSER](https://github.com/yuwchen/NRSER) for more details. (the paper is under double blind review ... so not available yet.)

[Pretrained model link](https://drive.google.com/drive/folders/12dTsiwFuPEu7n3tKJdSdko2-CfSvYlVz?usp=sharing)

```
python test_gpu.py --datadir /path/to/wav/dir --ckptdir emotion_model_v1_audioset-noise_model_v1_audioset-f16 #if you use gpu
python test_cpu.py --datadir /path/to/wav/dir --ckptdir emotion_model_v1_audioset-noise_model_v1_audioset-f16 #if you use cpu
```

# Step 5: Select audio files based on the SNR level detection score

- Keep audio files with SNR level score above a threshould (e.g.,>0.99).
- Mix original signals with enhanced signals accoreding to the SNR level score. 
For example:
```
S_re = S_ori x SNR_level_score + S_en x (1-SNR_level_score)
```
In idea case, for the clean file with a SNR_level_score of 1.0. The final wav file is the original signals, i.e, S_ori.
for the audio files only contain background noise (i.e, SNR_level_score of 0.0), the final wav file is only based on enhanced signals. 


