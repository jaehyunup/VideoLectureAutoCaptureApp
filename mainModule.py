import numpy
from PIL import ImageGrab
import skimage.measure as ski
import cv2

n=0
current_image = ImageGrab.grab(bbox=(0, 0, 1011, 760)) #화면 캡쳐
cv_img = cv2.cvtColor(numpy.array(current_image), cv2.COLOR_BGR2GRAY) #그레이 스케일 변환
privious_image = current_image
while(True):
    # 이미지 차영상을 위한 그레이스케일 변환
    # mat 형식의 imgGrab을 numpyarray 형식으로 전환 (출력하기위해)
    cv2.waitKey(1)
    cv_img = cv2.cvtColor(numpy.array(current_image), cv2.COLOR_BGR2GRAY)
    privious_image_gray = cv2.cvtColor(numpy.array(privious_image), cv2.COLOR_BGR2GRAY)

    (score, diff) = ski.compare_ssim(cv_img, privious_image_gray, full=True)
    #diff = (diff*255).astype("uint8")  # 반환받은 diff이미지 변환
    if( score < 0.88 ):
        pricapture=cv2.cvtColor(numpy.array(privious_image), cv2.COLOR_BGR2RGB) # 다시 RGB스케일로 변환
        cv2.imwrite('capture{}.jpg'.format(n), pricapture)
        print("화면이 캡쳐되었습니다 score:{}".format(score))
        n=n+1
    privious_image=current_image
    current_image = ImageGrab.grab(bbox=(0, 0, 1011, 760))  # 새로 화면을 캡쳐하고 현재화면으로 저장

cv2.destroyAllWindows()
