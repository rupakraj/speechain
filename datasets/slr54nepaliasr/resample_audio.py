# This will resample the audio to 16kHz
#        The dataset generator also does this but too slow

# this tool will use ffmpeg installed in the server and
# multiprocessing to speed up the process

import os
import subprocess
import multiprocessing


def resample_audio(input_file, output_file):
    # just show file name being processed
    print(f"Processing {input_file}")
    # ffmpeg in non verbose mode
    subprocess.call(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-i', input_file, '-ar', '16000', output_file])


if __name__ == '__main__':
    root_dir = os.getenv('SPEECHAIN_ROOT')
    input_dir  = os.path.join(root_dir, 'datasets', 'slr54nepaliasr', 'data', 'wav')
    output_dir = os.path.join(root_dir, 'datasets', 'slr54nepaliasr', 'data', 'wav_16k')

    os.makedirs(output_dir, exist_ok=True)
    # inside the wav there are subfolders also
    subfolders = os.listdir(input_dir)

    # iterate over all the subfolders and process them
    for subfolder in subfolders:
        input_subdir  = os.path.join(input_dir, subfolder)
        output_subdir = os.path.join(output_dir, subfolder)

        if not os.path.exists(output_subdir):
            os.makedirs(output_subdir, exist_ok=True)

        files = os.listdir(input_subdir)

        # make sure all the files are wav files
        files = [f for f in files if f.endswith('.wav')]

        # skip if already processed
        files = [f for f in files if not os.path.exists(os.path.join(output_subdir, f))]

        # if no files skip
        if len(files) == 0:
            print(f"Skipping {subfolder} as all files are already processed")
            continue

        # make sure cpu is not overloaded
        num_processes = min(multiprocessing.cpu_count() - 4, len(files))

        with multiprocessing.Pool(processes=num_processes) as pool:
            pool.starmap(resample_audio, [(os.path.join(input_subdir, f), os.path.join(output_subdir, f)) for f in files])

