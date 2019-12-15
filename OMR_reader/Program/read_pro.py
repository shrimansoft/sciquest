
import argparse
import cv2 as cv
import glob
import numpy as np
import pandas as pd
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



# Default mutable arguments should be harmless here
def draw_point(point, img, radius=5, color=(0, 0, 255)):
    cv.circle(img, tuple(point), radius, color, -1)




def get_alternative_patches(img_transf,sorted_contours): 
    return map(lambda c:img_transf[c[1]:c[1]+c[3],c[0]:c[0]+c[2]],sorted_contours)

    


def split(pach,n,orientation='h'): # orientation is either v or h
    "This function will spit the given image(pach)into n peices and cuting will\
     also consider wather it shoud be cute in to horizanto or vertical"
    if orientation == 'v' :
        sh=lambda c:int(np.round((pach.shape[0]*c)/n))
        for i in range(n):
            yield pach[sh(i):sh(i+1)]
    if orientation == 'h' :
        sh=lambda c:int(np.round((pach.shape[1]*c)/n))
        for i in range(n):
            yield pach[:,sh(i):sh(i+1)]
            




def get_marked_alternative(alternative_patches):
    "this function will get a list of samll images(array) and conpayer the contrast of therm\
    and cullect the one with with most high culler intancity"
    means = list(map(np.mean, alternative_patches))
    # print("hello test point")
    # print(means)
    sorted_means = sorted(means)
    # print("\n------------------\n")
    # print(np.argmin(means))
    # print("\n\n\n")

    # Simple heuristic

    if  means[np.argmin(means)]/means[np.argmax(means)] < 0.9:
        if sorted_means[0]/sorted_means[1] >0.9:
            return 55
        return np.argmin(means)
    else:
        return None




# def get_letter(alt_index,typ):
#     """alph -> alphabats \
#     num  -> numbers \
#     year -> year ( 1998-2004 ) \
#     opt -> 4 option A,B,C,D
#     gend ->gender male of female
#     std -> calls in which (8-12)"""
#     if typ == "alph":
#         return ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z'][alt_index] if alt_index is not None else np.nan
#     if typ == "num":
#         return [0,1,2,3,4,5,6,7,8,9][alt_index] if alt_index is not None else np.nan
#     if typ == "year":
#         return [1998,1999,2000,2001,2002,2003,2004][alt_index] if alt_index is not None else np.nan
#     if typ == "opt":
#         return ["A", "B", "C", "D", "E"][alt_index] if alt_index is not None else np.nan
#     if typ == "gend":
#         return ["MALE","FEMALE","OTHER"][alt_index] if alt_index is not None else np.nan
#     if typ == "std":
#         return [8,9,10,11,][alt_index] if alt_index is not None else np.nan
#     if typ == "sub" :
#         return ["Matsh","Biology"][alt_index] if alt_index is not None else np.nan


def recognize(img_transf,countours,n,orientation='h'):
    pachs=get_alternative_patches(img_transf,countours)
    answers = []
    for i,pach in enumerate(pachs):
        # print("-----",i+1,"------")
        # cv.imshow("test",pach)
        # cv.waitKey()
        splited_pach=split(pach,n,orientation)
        # for i in splited_pach:
            # cv.imshow("test2",i)
            # cv.waitKey()
        alt_index=get_marked_alternative(splited_pach)
        # print(alt_index)
        answers.append(alt_index)
    return answers



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
    File= open("./Config/cutters.data","r")    
    cuttersRead = File.readlines()
    # print(cuttersRead)
    File.close()

    
    
    cropedImage= cv.imread(args.input,0)

    File= open(("./readedData/" + args.input.split("/")[-1].split(".")[0]),"a+")
    for i in cuttersRead:
        print('*** ---------------------------------- ***')
        print(i)
        cutter = ast.literal_eval(i)
        contorus = cutter[-1]
        name = cutter[0]
        num = int(cutter[1])
        print("num: ",num)
        orian = cutter[2]
        print("orian: ",orian)
        print("a : ",cutter[3])
        print("b : ",cutter[4])


        answer=recognize(cropedImage,contorus,num,orian)
        File.write(str([name,answer])+'\n')
    File.close()


    # lists = ast.literal_eval(strin

    # reading image 
   
    

    
    
    #------------------Student biodata-----------------------------------
    
    
    # data=[rollNO,Std,Sub,DOB,Date,Gender,MobiNum]
    # print(data)
    
    # omr_data=pd.read_csv("data.csv",index_col=0)
    
    # omr_data.loc[len(omr_data)]=data
    
    
    # omr_data.to_csv("data.csv")
    
    # #------------------------storing marks-----------------------------
    
    
    # data=[rollNO]+answer
    # print(data)
    
    # omr_ans=pd.read_csv("ans.csv",index_col=0)
    # print(omr_ans)
    
    # omr_ans.loc[len(omr_ans)]=data
    
    
    # omr_ans.to_csv("ans.csv")
    
    # cv.waitKey()


if __name__ == '__main__':
    main()
