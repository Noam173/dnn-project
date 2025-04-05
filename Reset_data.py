import shutil
import Data_Manipulation

path = Data_Manipulation.create_directory()
shutil.rmtree(f"{path}")
