# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import shutil
import os
import cudf as pd
import Data_Manipolation
data=Data_Manipolation.create_directory()
shutil.rmtree(f'{data}')



