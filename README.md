# perspective-warp

Basic Homography transformation to project ground back to camera plane.

This also uses optical tracking on a few points to be used for vehicle speed estimation. 

Dependencies:
----------------
- opencv : install opencv 3
- libterraclear : clone from https://github.com/TerraClear/libterraclear
- darknet : clone and build (for CUDA, CUDNN, OPENCV) from https://github.com/TerraClear/darknet
        *Note: create symbolic link for libdarknet.so to /usr/local/lib/libdarknet.so


![alt text](https://i.imgur.com/vx2FOa2.png)
