/*
 * Basic code to correct camera ground projection into a pinhole view
 * and also track via optical tracking..
 * Copyright (C) 2019 Jacobus du Preez
 * 
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 * 
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 * 
 * CREATED BY: Koos du Preez - koos@terraclear.com
 * 
*/

#include <iostream>

//OpenCV
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"

//YOLO V3 INCLUDES - https://github.com/AlexeyAB/darknet
//compiled for SO/Lib and OPENCV using GPU(Cuda/CuDNN) and GPU Tracking.
#define OPENCV
#define TRACK_OPTFLOW
#define GPU
#include <darknet/include/yolo_v2_class.hpp>


//#ifndef TC_USE_BLACKFLY
//    #define TC_USE_BLACKFLY
//#endif
//#include "libterraclear/src/camera_flir_blackfly.hpp"

#include "libterraclear/src/camera_async.hpp"
#include "libterraclear/src/camera_file.hpp"
#include "libterraclear/src/vision_warp.h"
namespace tc = terraclear;

using namespace std;
using namespace cv;

bool get_tracked_anchor(std::vector<bbox_t> &bbox_list, bbox_t &anchor)
{
    bool retval = false;
    
    for (auto bbox : bbox_list)
    {
        //found it.. update anchor..
        if (bbox.track_id == anchor.track_id)
        {
            //update anchor..
            anchor.x = bbox.x;
            anchor.y = bbox.y;
            anchor.w = bbox.w;
            anchor.h = bbox.h;
            anchor.frames_counter = bbox.frames_counter;
            
            retval = true;
            
            break;                 
        }
    }
    
    return retval;
}

int main(int argc, char** argv)
{
    //** OPEN CV CUDA TRACKING
    //DEFAULTS: Tracker_optflow tracker_engine(int _gpu_id = 0, int win_size = 9, int max_level = 3, int iterations = 8000, int _flow_error = -1)
    Tracker_optflow tracker_engine(0, 21, 6, 8000, -1);

    //video fps
    uint32_t fps = 60;
    
    //create file_cam
    tc::camera_file cam_base("/home/koos/Downloads/idaho-videos/camera_3.mp4");
    cam_base.update_frames();

    tc::camera_async cam_async(&cam_base, fps, true);
    cam_async.thread_start("async_cam");
    
    //source img
    cv::Size src_size(1440, 1080);
    Mat src_img = cam_async.get_ImageBuffer();

    //destination img
    cv::Size dst_size(700,900);
    cv::Mat dst_img(dst_size, CV_8UC3);

    //warp transform tool
    tc::vision_warp warp_transform;

    //adjust source Top Left and Right
    warp_transform._source_points.top_left.x = 450;
    warp_transform._source_points.top_right.x = 1000;
    warp_transform._source_points.top_left.y = warp_transform._source_points.top_right.y = 380;

    //adjust source Bottom Left and Right
    warp_transform._source_points.bottom_left.x = 0;
    warp_transform._source_points.bottom_right.x = 1440;
    warp_transform._source_points.bottom_right.y = warp_transform._source_points.bottom_left.y = src_size.height;
    
    //set target size
    warp_transform._target_size = dst_size;
    
    //init and pre-calc transformation matrix
    warp_transform.init_transform();

    //speed tracking points and settings
    uint32_t    track_start_y = 500;
    uint32_t    track_end_y = 900;
    uint32_t    track_max_travel = 100;
    uint32_t    track_offset_y = 40;
    uint32_t    track_xy_size = 40;
    
    uint32_t    track_count = (track_end_y - track_start_y) / (track_max_travel - track_offset_y);
    uint32_t    track_offset_x = dst_size.width / (track_count + 1);
    uint32_t    box_x = track_offset_x;
    uint32_t    box_y = track_start_y;
    
    //vector of anchor boxes, keys by ID
    std::vector<bbox_t> track_anchors;
    for (uint32_t t = 0; t < track_count; t++ )
    {
        bbox_t tmp_box;
        tmp_box.x = box_x - (track_xy_size / 2);
        tmp_box.y = box_y - (track_xy_size / 2);
        tmp_box.w = tmp_box.h = track_xy_size;
        tmp_box.track_id = t;
        
        //keep track of anchors by ID
        track_anchors.push_back(tmp_box);
        
        box_x += track_offset_x;
        box_y += track_offset_y;
    }

    
    //source and target windows and auto resize.
    std::string src_window = "Source";
    namedWindow(src_window, WINDOW_NORMAL);
    cv::resizeWindow(src_window, src_size);
    
    std::string dst_window = "Target";
    namedWindow(dst_window, WINDOW_NORMAL);
    cv::resizeWindow(dst_window, dst_size);

    //vector of tracked boxes.
    std::vector<bbox_t> track_boxes;
    
    //start with anchor boxes
    tracker_engine.update_cur_bbox_vec(track_anchors);


    bool paused = false;
    while (true)
    {
        if (src_img.rows > 0)
        {
            //warp original & resize.
            dst_img = warp_transform.transform_image(src_img);
            
            //do tracking of boxes
            track_boxes = tracker_engine.tracking_flow(dst_img);
            
            //ensure anchors were tracked and not lost or past limits..
            std::vector<bbox_t> track_boxes_new;
            for (auto anchor : track_anchors)
            {
                //get anchor, check if tracked..
                bbox_t tmp_bbox = anchor;
                if (get_tracked_anchor(track_boxes, tmp_bbox))
                {
                    //if anchor tracked and not past travel limits, keep tracking
                    //else reset back to anchor.. ..
                    if ((tmp_bbox.y + track_xy_size / 2) < (anchor.y + track_max_travel))
                    {
                        //keep tracked box
                        track_boxes_new.push_back(tmp_bbox);
                    }
                    else
                    {
                        //reset to anchor..
                        track_boxes_new.push_back(anchor);
                    }
                        
                }
                else
                {
                    //add un-tracked or lost anchor..
                    track_boxes_new.push_back(anchor);
                }
                
                //update tracker engine with corrected box positions..
                tracker_engine.update_cur_bbox_vec(track_boxes_new);
                
                //draw the anchor start and end lines
                cv::Point bbox_start(anchor.x + anchor.h / 2, anchor.y + anchor.w / 2);
                cv::Point bbox_end(bbox_start.x, bbox_start.y + track_max_travel);
                line(dst_img, bbox_start, bbox_end, Scalar(0, 0xff, 0x00), 2);

                //draw tracked areas..
                cv::Rect bbox_rect;
                bbox_rect.x = tmp_bbox.x;
                bbox_rect.y = tmp_bbox.y;
                bbox_rect.width = tmp_bbox.w;
                bbox_rect.height = tmp_bbox.h;
                cv::rectangle(dst_img, bbox_rect, cv::Scalar(0x00, 0xff, 0xff), 2);
            }

            //draw lines for warp sources.
            line(src_img, warp_transform._source_points.top_left,warp_transform._source_points.top_right, Scalar(0, 0, 255), 2);
            line(src_img, warp_transform._source_points.top_right, warp_transform._source_points.bottom_right, Scalar(0, 0, 255), 2);
            line(src_img, warp_transform._source_points.bottom_right, warp_transform._source_points.bottom_left, Scalar(0, 0, 255), 2);
            line(src_img, warp_transform._source_points.bottom_left,warp_transform._source_points.top_left, Scalar(0, 0, 255), 2);
            

            //if paused, display "paused"
            if (paused)
            {
                std::string str_paused = "PAUSED ||";
                cv::putText(src_img, str_paused, cv::Point(10, 50), CV_FONT_HERSHEY_PLAIN, 2, cv::Scalar(0x00, 0x00, 0xff), 2);
                cv::putText(dst_img, str_paused, cv::Point(10, 50), CV_FONT_HERSHEY_PLAIN, 2, cv::Scalar(0x00, 0x00, 0xff), 2);
            }
            
            //show src and dst windows
            imshow(src_window, src_img);
            imshow(dst_window, dst_img);
        }
        

        //get next frame..
        src_img = cam_async.get_ImageBuffer();

        //wait for keys
        int x = waitKey(15);
        if (x == 27) //ESCAPE = exit..
        {
            break;
        }
        if (x == 112) //SPACE = pause..
        {
            if (!paused)
            {
                cam_async.thread_pause();
                paused = true;
            }
            else
            {
                cam_async.thread_resume();
                paused = false;
            }
        }
    }
    
    cam_async.thread_stopwait();
    
    return 0;
}

