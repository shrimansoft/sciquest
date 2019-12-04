import argparse
import cv2 as cv
import glob
import numpy as np

CORNER_FEATS = (
    0.322965313273202,
    0.19188334690998524,
    1.1514327482234812,
    0.998754685666376,
)
TRANSF_SIZE = 512
sheet_size=[1604,2270]



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
    cv.imwrite("./Croped/"+args.input.split("/")[-1],cropedImage,[cv.IMWRITE_PNG_COMPRESSION])
    

if __name__ == '__main__':
    main()
