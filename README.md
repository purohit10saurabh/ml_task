Main Dependencies - Torch, python3

Make sure /data folder (which has /recorded_images and /actions.npy) is in same directory as /Task (created after unzipping)

Works only with atleast 2 gpus with device ids cuda:0,1

Run in terminal-

cd Task/code
python3 main.py

Done
--------------------------------------------------------------------------------------------------------

Bonus Part-
Main Dependencies - Libtorch, CMake, gcc

Run in terminal-

cd Task/code
bash run_port.sh
