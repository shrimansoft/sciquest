import argparse
import cv2 as cv
import glob
import numpy as np
import ast

CORNER_FEATS = (
    0.322965313273202,
    0.19188334690998524,
    1.1514327482234812,
    0.998754685666376,
)
TRANSF_SIZE = 512
sheet_size=[1604,2270]


def get_approx_contour(contour, tol=.01):
    """Get rid of 'useless' points in the contour"""
    epsilon = tol * cv.arcLength(contour, True)
    return cv.approxPolyDP(contour, epsilon, True)


def get_contours(image_gray):
    im2, contours, hierarchy = cv.findContours(
        image_gray, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    #print(contours)
    #cv.imshow("im2",im2)
    #cv.imwrite("test2.png",im2,[cv.IMWRITE_PNG_COMPRESSION])

    return map(get_approx_contour, contours)



def get_alternative_contoures(cuter,a,b):  #tan(thita)=a/b
    "this function will take two image file of same size one to cut in peices\
    or the working image (img_transf) another one is having black spots where to cut this things\
    a and b are for knowing the value of tan(thita)=a/b and this line will chose which will have to come first"
    grey= cuter
    ret, im = cv.threshold(grey, 127, 255, cv.THRESH_BINARY)
    #con=list(get_contours(im))
    #cv.drawContours(cuter, con, -1, (0, 255, 0), 3)            
    #cv.imshow("shri",cuter)
    print("testPoint2")
    #cv.waitKey()
    rect_contours=map(cv.boundingRect,get_contours(im))
    apro_contours=(i for i in rect_contours if i[2]*i[3]<3000000)
    sorted_contours=sorted(apro_contours,key=lambda c:c[0]*a+c[1]*b)
    return sorted_contours

    







def main():

    #--------------------- taking file name form argument ---------------------------------------
    parser= argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        help="Input image filename",
        required=True,
        type=str)

    args = parser.parse_args()
    print(args.input)
    argsAry=args.input.split("/")[-1].split(".")[0].split("_")
    print("testpint 1")
    name = argsAry[0]
    num = argsAry[1]
    orian = argsAry[2]
    a = argsAry[3]
    b = argsAry[4]
    print(argsAry)


    # reading image 
    originalCoutter= cv.imread(args.input,0)
    print("testPoint3")
    cuttersPoint=get_alternative_contoures(originalCoutter,a,b)
    
    strin = str([name,num,orian,a,b,cuttersPoint])
    print(strin)

    File= open("./Config/cutters.data","a+")    
    File.write(strin+'\n')
    File.close()



    # lists = ast.literal_eval(strin)
    # print(type(lists[2]))
    # print(lists)

    #cv.imshow("imput image",im_org)
    #cv.waitKey()

   
    
    
    

    



if __name__ == '__main__':
    main()
