#!/bin/sh
cd /CV_team_data_01/lyp_data/weights/b/models/DCNv2;
python setup.py build develop;
cd /CV_team_data_01/lyp_data/weights/b;
python test_widerface_ms150.py;
