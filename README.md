### Hello, this is a personal project named Arrangeit, which is constructed to assist craftpeople automatically arranging and packing patterns of mechanism parts for laser-cutting or milling.


## NOTICE
# The project is mainly consisted by two parts: 'data' and 'src'. The scripts under the folder 'module' are just some trials of mine in order to varify the reliability of ezdxf library.
# There is no GUI for this project so far, make sure you get all libraries prepared before you try it out.


## WORKFLOW
# 1. Create you CAD file which contains a single copy of each template (without the frame of the substrate board) and save it as .dxf file in 'data/input'. Remember to rename it as completeINPUT.dxf.

# 2. The dxf_processor will load the completeINPUT.dxf flie within 'data/input', recognize every single part, pack up all the lines and curves (including the holes) that belongs to each template in to editable blocks, and finally produce an image listing each templates with their ID numbers which is an important reference for further data input.

# 3. Write your own input file that clarifies how many copies of each template do you want and the range of rotation angles allowed. You should create the excel list in the format below and save it in 'data/intermediate' as 'user_input'.

# NO TLTLES NOR COLUME/ROW LABLE!!!
#          #               #    #    #    #
#   number of copies       0    90   180  270
#         ...            (if this angle is allowed type it in the box, increase from left to right)
#         ...

# 4. After the template_matrix folder within 'data/intermediate' is filled with .npy arrays, run main.py to start the optimization, and the module computation_core can be activated automatically.

# 5. The visualize_substrate.py is aimed to show you the result of the arrangement in the form of .png, only for tuning process. You can still run this script when the work is done to visualize the details of the arrangement just for a duble check.

# 6. The arranged CAD file is named 'transformed_parts.dxf' and saved in 'data/output'.


## REFERENCE
# Meng, L., Ding, L., Pu, Y. et al.
# Optimizing 2D irregular packing via image processing and computational intelligence. Sci Rep 15, 12320 (2025).
# https://doi.org/10.1038/s41598-025-97202-0