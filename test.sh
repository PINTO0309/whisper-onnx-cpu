#!/bin/bash

start_time=`date +%s`

python whisper/transcribe.py carmack.mp3 --model tiny.en --beam_size 3

end_time=`date +%s`

run_time=$((end_time - start_time))

echo $run_time