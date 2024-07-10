import cv2
import matplotlib.pyplot as plt
img = cv2.imread('temp.jpg')
gimg = cv2.imread('temp.jpg',cv2.IMREAD_GRAYSCALE)
display_width, display_height = 1080, 720  # Replace with your display resolution

# Resize the image to match the display resolution
img = cv2.resize(img, (display_width, display_height))
gimg = cv2.resize(gimg, (display_width, display_height))
cv2.imshow('BK',img)
cv2.waitKey(0)
cv2.imshow('BK',gimg)
cv2.waitKey(0)

cv2.destroyAllWindows()