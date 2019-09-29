import argparse
import cv2 as cv
import glob
import numpy as np
import pandas as pd

CORNER_FEATS = (
    0.322965313273202,
    0.19188334690998524,
    1.1514327482234812,
    0.998754685666376,
)
TRANSF_SIZE = 512
sheet_size=[1604,2270]



if not glob.glob("ans.csv"):
    columns=["RollNo"]+list(range(1,81))
    omr_ans=pd.DataFrame(columns=columns)
    omr_ans.to_csv("ans.csv")
if not glob.glob("data.csv"):
    columns=["RollNo","STD","Subject","DOB","Date","Gender","MobileNum"]
    omr_data=pd.DataFrame(columns=columns)
    omr_data.to_csv("data.csv")

def corner_point_contour(contour):
    "this function will find the corner point of contour(L shaped) by \
    first find the points with longest distance and then drow a line from \
    this two points now which point is farest from the line is the corner of \
    the contour"
    p1=0
    p2=1
    line=(p1,p2)
    for i in range(6):
        for j in range(i+1,6):
            if(np.linalg.norm(contour[i]-contour[j])>np.linalg.norm(contour[p1]-contour[p2])):
                p1=i
                p2=j
            #print(i,j)
            #print((contour[i],contour[j]))
            #print((contour[i]-contour[j]))
            #print(np.linalg.norm(contour[i]-contour[j]))
    print(p1,p2)
    line=[p1,p2]
    exce=[i for i in range(6) if i not in line]
    print(exce)
    return contour[sorted(exce,key=lambda c:np.linalg.norm(np.cross(contour[p2]-contour[p1], contour[p1]-contour[c]))/np.linalg.norm(contour[p2]-contour[p1]),reverse=True)[0]][0]
    


def normalize(im):
    return cv.normalize(im, np.zeros(im.shape), 0, 255, norm_type=cv.NORM_MINMAX)

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

def get_corners(contours):
    return sorted(
        contours,
        key=lambda c: features_distance(CORNER_FEATS, get_features(c)))[:4]

def get_bounding_rect(contour):
    rect = cv.minAreaRect(contour)
    box = cv.boxPoints(rect)
    return np.int0(box)

def get_convex_hull(contour):
    return cv.convexHull(contour)

def get_contour_area_by_hull_area(contour):
    return (cv.contourArea(contour) /
            cv.contourArea(get_convex_hull(contour)))

def get_contour_area_by_bounding_box_area(contour):
    return (cv.contourArea(contour) /
            cv.contourArea(get_bounding_rect(contour)))

def get_contour_perim_by_hull_perim(contour):
    return (cv.arcLength(contour, True) /
            cv.arcLength(get_convex_hull(contour), True))

def get_contour_perim_by_bounding_box_perim(contour):
    return (cv.arcLength(contour, True) /
            cv.arcLength(get_bounding_rect(contour), True))

def get_features(contour):
    try:

        print(
            get_contour_area_by_hull_area(contour),
            get_contour_area_by_bounding_box_area(contour),
            get_contour_perim_by_hull_perim(contour),
            get_contour_perim_by_bounding_box_perim(contour),
        )
        return (
            get_contour_area_by_hull_area(contour),
            get_contour_area_by_bounding_box_area(contour),
            get_contour_perim_by_hull_perim(contour),
            get_contour_perim_by_bounding_box_perim(contour),
        )
    except ZeroDivisionError:
        return 4*[np.inf]

def features_distance(f1, f2):
    return np.linalg.norm(np.array(f1) - np.array(f2))

# Default mutable arguments should be harmless here
def draw_point(point, img, radius=5, color=(0, 0, 255)):
    cv.circle(img, tuple(point), radius, color, -1)

def get_centroid(contour):
    m = cv.moments(contour)
    x = int(m["m10"] / m["m00"])
    y = int(m["m01"] / m["m00"])
    return (x, y)

def order_points(points):
    """Order points counter-clockwise-ly."""
    origin = np.mean(points, axis=0)

    def positive_angle(p):
        x, y = p - origin
        ang = np.arctan2(y, x)
        return 2 * np.pi + ang if ang < 0 else ang

    return sorted(points, key=positive_angle)

def get_outmost_points(contours):
    return list(map(corner_point_contour,contours))

def perspective_transform(img, points):
    "Transform img so that points are the new corners"
    source = np.array(
        points,
        dtype="float32")

    dest = np.array([
        sheet_size,
        [0, sheet_size[1]],
        [0, 0],
        [sheet_size[0], 0]],
        dtype="float32")

    img_dest = img.copy()
    transf = cv.getPerspectiveTransform(source, dest)
    warped = cv.warpPerspective(img, transf, tuple(sheet_size))
    return warped

def get_alternative_patches(img_transf,cuter,a,b):  #tan(thita)=a/b
    "this function will take two image file of same size one to cut in peices\
    or the working image (img_transf) another one is having black spots where to cut this things\
    a and b are for knowing the value of tan(thita)=a/b and this line will chose which will have to come first"
    grey= cuter
    ret, im = cv.threshold(grey, 127, 255, cv.THRESH_BINARY)
    con=list(get_contours(im))
    cv.drawContours(cuter, con, -1, (0, 255, 0), 3)
                 
    #cv.imshow("shri",cuter)
    #cv.waitKey()
    rect_contours=map(cv.boundingRect,get_contours(im))
    apro_contours=(i for i in rect_contours if i[2]*i[3]<3000000)
    sorted_contours=sorted(apro_contours,key=lambda c:c[0]*a+c[1]*b)
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
    # print(means)
    sorted_means = sorted(means)
    #print("\n------------------\n")
    #print(np.argmin(means))
    #print("\n\n\n")

    # Simple heuristic

    if sorted_means[0]/sorted_means[1] >0.9:
        return None
    return np.argmin(means)

def draw_marked_alternative(question_patch, index):
    cx, cy = sheet_coord_to_transf_coord(
        50 * (2 * index + .5),
        50/2)
    draw_point((cx, TRANSF_SIZE - cy), question_patch, radius=5, color=(255, 0, 0))
    #cv.imshow("test",question_patch)
    #cv.waitKey()




def get_letter(alt_index,typ):
    """alph -> alphabats \
    num  -> numbers \
    year -> year ( 1998-2004 ) \
    opt -> 4 option A,B,C,D
    gend ->gender male of female
    std -> calls in which (8-12)"""
    if typ == "alph":
        return ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z'][alt_index] if alt_index is not None else np.nan
    if typ == "num":
        return [0,1,2,3,4,5,6,7,8,9][alt_index] if alt_index is not None else np.nan
    if typ == "year":
        return [1998,1999,2000,2001,2002,2003,2004][alt_index] if alt_index is not None else np.nan
    if typ == "opt":
        return ["A", "B", "C", "D", "E"][alt_index] if alt_index is not None else np.nan
    if typ == "gend":
        return ["MALE","FEMALE","OTHER"][alt_index] if alt_index is not None else np.nan
    if typ == "std":
        return [8,9,10,11,][alt_index] if alt_index is not None else np.nan
    if typ == "sub" :
        return ["Matsh","Biology"][alt_index] if alt_index is not None else np.nan


def recognize(img_transf,cuter,a,b,n,typ,orientation='h'):
    pachs=get_alternative_patches(img_transf,cuter,a,b)
    answers = []
    for i,pach in enumerate(pachs):
        # print("-----",i+1,"------")
        # cv.imshow("test",pach)
        # cv.waitKey()
        splited_pach=split(pach,n,orientation)
        #for i in splited_pach:
            #cv.imshow("test2",i)
            #cv.waitKey()
        alt_index=get_marked_alternative(splited_pach)
        # print(alt_index)
        answers.append(get_letter(alt_index,typ))
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



    # reading image 
    originalImage= cv.imread(args.input,0)
    #cv.imshow("imput image",im_org)
    #cv.waitKey()

    scale_percent = 40 # percent of original size
    width = int(originalImage.shape[1] * scale_percent / 100)
    height = int(originalImage.shape[0] * scale_percent / 100)
    dim = (width, height)
	# resize image
    im_org = cv.resize(originalImage, dim, interpolation = cv.INTER_AREA)

    
    blurred = cv.GaussianBlur(im_org,(11,11),10)
  
    
    #cv.imshow("blurred",blurred)
    
    
    
    im = normalize(blurred)
    
    
    
    #cv.imshow("im",im)
    
    ret, im = cv.threshold(im, 127, 255, cv.THRESH_BINARY)
    # cv.imshow("im",im)
    
    
    contours = get_contours(im)
    corners = get_corners(contours)
    cv.drawContours(im_org, corners, -1, (0, 255, 0), 3)
    
    #cv.imshow("im_orig2",im_org)
    
    
    outmost = order_points(get_outmost_points(corners))
    
    # for i in outmost:
        # draw_point(i, im_org, radius=5, color=(255, 0, 0))
    
    cropedImage = perspective_transform(im_org, outmost)
    cv.imwrite("../output_data/"+args.input[14:],cropedImage,[cv.IMWRITE_PNG_COMPRESSION])
    
    #cv.imshow("transf",transf)
    
    #---loding cutter and taking data of answer-------
    cutter= cv.imread("cutter2/answer_11.png",0)
    answer=recognize(cropedImage,cutter,1143,100,4,'opt','h')
    
    #---loding cutter and taking data of date 2019-------
    cutter= cv.imread("cutter2/Date(2019).png",0)
    date=recognize(cropedImage,cutter,1143,100,10,'num','v')
    
    
    #---loding cutter and taking data of DOB_DD-MM-------
    cutter= cv.imread("cutter2/DOB_DD-MM.png",0)
    DOB_DD=recognize(cropedImage,cutter,1143,100,10,'num','v')
    
    #---loding cutter and taking data of DOB_year-------
    cutter= cv.imread("cutter2/DOB_year.png",0)
    DOB_year=recognize(cropedImage,cutter,1143,100,7,'year','v')
    
    #---loding cutter and taking data of gender------
    cutter= cv.imread("cutter2/gender.png",0)
    gander=recognize(cropedImage,cutter,1143,100,3,'gend','v')
    
    #---loding cutter and taking data of mobile_number-------
    cutter= cv.imread("cutter2/mobile_number.png",0)
    mobile_no=recognize(cropedImage,cutter,1143,100,10,'num','v')
    
    
    #---loding cutter and taking data of rollno-aplha-------
    cutter= cv.imread("cutter2/rollno_alph.png",0)
    roll_alpha=recognize(cropedImage,cutter,1143,100,26,'alph','v')
    
    #---loding cutter and taking data of rollno_num-------
    cutter= cv.imread("cutter2/rollno_num.png",0)
    rollno_num=recognize(cropedImage,cutter,1143,100,10,'num','v')
    
    #---loding cutter and taking data of STD -------
    cutter= cv.imread("cutter2/std.png",0)
    std=recognize(cropedImage,cutter,1143,100,4,'std','v')
    
    
    #---loding cutter and taking data of subject -------
    cutter= cv.imread("cutter2/subject.png",0)
    subject=recognize(cropedImage,cutter,1143,100,2,'sub','v')
    
    DOB="{}/{}/{}".format(DOB_DD[0]*10+DOB_DD[1],DOB_DD[2]*10+DOB_DD[3],DOB_year[0])
    Date="{}/{}/{}".format(date[0]*10+date[1],date[2]*10+date[3],"2019")
    rollNO="{}{}{}{}{}{}{}{}".format(roll_alpha[0],roll_alpha[1],roll_alpha[2],rollno_num[0],rollno_num[1],rollno_num[2],rollno_num[3],rollno_num[4])
    MobiNum="".join(map(str,mobile_no))
    Gender=str(gander[0])
    Std=str(std[0])
    Sub=str(subject[0])
    
    
    #------------------Student biodata-----------------------------------
    
    
    data=[rollNO,Std,Sub,DOB,Date,Gender,MobiNum]
    print(data)
    
    omr_data=pd.read_csv("data.csv",index_col=0)
    
    omr_data.loc[len(omr_data)]=data
    
    
    omr_data.to_csv("data.csv")
    
    #------------------------storing marks-----------------------------
    
    
    data=[rollNO]+answer
    print(data)
    
    omr_ans=pd.read_csv("ans.csv",index_col=0)
    print(omr_ans)
    
    omr_ans.loc[len(omr_ans)]=data
    
    
    omr_ans.to_csv("ans.csv")
    
    cv.waitKey()


if __name__ == '__main__':
    main()
