## Overview
This is the repository for the DExScore tool, a tool for eliciting domain expertise from users. Currently, this tool is configured to work to build programs for two tasks: an image classification task asking "Is this image one of people playing sports?" and a music querying task "Is this piece of music similar to other positively recommended pieces of music?".

## Prerequisites
DExScore requires Python to run. Additionally, to manually run the music task, [PyGame](https://www.pygame.org/wiki/GettingStarted) is required to play sounds. This can be installed with `pip install pygame`. 

## Manually running
To run DExScore manually, use the scripts `manual_sports.sh` and `manual_music.sh` to run the sports and music tasks, respectively. `manual_music.sh` will error if Pygame is not installed.

## Evaluations
To reproduce the evaluations described in our paper, run `eval_all.sh`. This will take several hours to run. To reproduce all the plots but with a smaller sample size, run `eval_small.sh`