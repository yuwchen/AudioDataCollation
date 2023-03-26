import os
import pandas as pd
from tqdm import tqdm
import librosa
import soundfile as sf
import re
import argparse

WAV_LENGTH = 0.5 #save wave files that longer than 0.5s

def cut_wav(df, wav_dir, output_dir, output_csv):

    
    wav_list = list(df.wavname.unique())
    column_names = list(df.columns.values)
    column_names.append('wavname_seg')
    new_test_df = pd.DataFrame(columns=column_names)
    
    for wav_name in tqdm(wav_list):
        the_df = df.loc[df['wavname'] == wav_name]
        y, sr = librosa.load(os.path.join(wav_dir, wav_name))
        wav_name_seg = []
        for _, row in the_df.iterrows():
            start_time = row['start']
            end_time = row['end']
            wav_length = end_time-start_time
            start_point = int(start_time*sr)
            end_point = int(end_time*sr)
            output_name = wav_name.replace('.wav','') +'_'+str(start_point)+'_'+str(end_point)
            transcript = row['transcript']
            
            if not re.search('[a-zA-Z]', transcript) or (transcript.isalpha()): 
                wav_name_seg.append('Nan')
            else:
                if wav_length>WAV_LENGTH:
                    wav_name_seg.append(output_name)
                    outputpath = os.path.join(output_dir, output_name+'.wav')
                    sf.write(outputpath, y[start_point:end_point], sr)
                else:
                    wav_name_seg.append('Nan')

        the_df['wavname_seg'] = wav_name_seg
        new_test_df = pd.concat([new_test_df, the_df], ignore_index = True) 

    new_test_df.to_csv(output_csv, index=False)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--datadir', default='./data',  type=str, help='Path of your DATA/ directory')
    parser.add_argument('--input_file', default='data_whisper.csv',  type=str, help='Path of the whisper segmentation result')

    args = parser.parse_args()
    wav_dir= args.datadir
    input_file= args.input_file

    df = pd.read_csv(input_file)
    output_dir = input_file.replace('.csv','_seg')

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    cut_wav(df, wav_dir, output_dir, input_file.replace('.csv','_seg.csv'))

if __name__ == '__main__':
    main()