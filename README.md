# AudioDataCollation

## Step 1: downloading audio files

Download from youtube:
- [youtube-dl](https://github.com/ytdl-org/youtube-dl) 
- [yt-dlp](https://github.com/yt-dlp/yt-dlp)


## Step 2: audio segmentation

### Step 2-1: using whisper to detect the audio segments 
```
whisper_asr.py --datadir /path/to/your/wav/file/dir
```
This will save the results in {dir}_whisper.csv

### Step 2-2: using whisper results to do the wavfile segmentation

```
cut_wav.py --datadir /path/to/your/wav/file/dir --input_file /path/to/your/{dir}_whisper.csv
```

This will save the wav segments in /path/to/your/wav/file/{dir}_seg


## Step 3: speech enhancement

Download code from [NRSER](https://github.com/yuwchen/NRSER).

Remove background noise of the audio.

```
python CMGAN/enhanced_speech_cpu.py --test_dir /path/to/wavfiles/dir #if you use cpu
python CMGAN/enhanced_speech_gpu.py --test_dir /path/to/wavfiles/dir #if you use gpu
```

The enhanced signals will be saved in the ./data/{dir}_en directory.

See [CMGAN](https://github.com/ruizhecao96/CMGAN) for more details. 

## Step 4: SNR level detection

Although speech enhancement can remove background noise from a noisy signal, it also causes distortion. Such distortion is especially bad for a high Signal-to-noise ratio (SNR) signal because these signals don't have background noise; thus, speech enhancement will only cause a negative impact. One possible method to address this issue is using an SNR-level detection model. 
The following model takes speech signals and the corresponding enhanced signals as input. Then, it outputs the SNR level score, which is in the range of 0-1. 

[Pretrained model link](https://drive.google.com/drive/folders/12dTsiwFuPEu7n3tKJdSdko2-CfSvYlVz?usp=sharing)

```
python test_gpu.py --datadir /path/to/wav/dir --ckptdir emotion_model_v1_audioset-noise_model_v1_audioset-f16 #if you use gpu
python test_cpu.py --datadir /path/to/wav/dir --ckptdir emotion_model_v1_audioset-noise_model_v1_audioset-f16 #if you use cpu
```

The results would be saved in {model_name}\_{data_dir}.txt
Output format:
```
{wavname};{predicted_emotion};A:{arousal};V:{valence};D:{dominance},{[probability of emotion category]},{[[SNR-level score]]}
```

Note that this SNR level detection model was designed based on the idea that "the original and enhanced signals of high SNR signals are similar because high SNR signals don't have much background noise to be removed." Although the experimental results show that the model can distinguish speech signals with different SNR levels, it fails when the speech enhancement model cannot remove the background noise. 


# Step 5: Select audio files based on predicted SNR level score

- Keep audio files with SNR level score above a threshould (e.g.,>0.99)

- Mix original signals with enhanced signals accoreding to the SNR level score. 

