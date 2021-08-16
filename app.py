from matplotlib.backends.backend_agg import RendererAgg
import time
import streamlit as st
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2

matplotlib.use("agg")
_lock = RendererAgg.lock
st.set_page_config(layout="wide")

def show_image(image):
    fig2 = plt.figure()
    image = mpimg.imread(image)
    plt.imshow(image)
    plt.show()
    st.pyplot(fig2)

def detect_lanes(image, low_threshold, high_threshold, threshold, min_line_length, rho, max_line_gap):
    fig2 = plt.figure()
    image = mpimg.imread(image)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # PARAMETERS
    kernel_size = 5
    ignore_mask_color = 255
    theta = np.pi / 180

    #CALCULATED FIELDS
    blur_gray = cv2.GaussianBlur(gray,(kernel_size, kernel_size),0)
    edges = cv2.Canny(blur_gray, low_threshold, high_threshold)
    mask = np.zeros_like(edges)
    imshape = image.shape
    vertices = np.array([[(0,imshape[0]),(450, 290), (490, 290), (imshape[1],imshape[0])]], dtype=np.int32)
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    masked_edges = cv2.bitwise_and(edges, mask)
    line_image = np.copy(image)*0
    lines = cv2.HoughLinesP(masked_edges, rho, theta, threshold, np.array([]),
                                min_line_length, max_line_gap)

    for line in lines:
        for x1,y1,x2,y2 in line:
            cv2.line(line_image,(x1,y1),(x2,y2),(255,0,0),10)

    color_edges = np.dstack((edges, edges, edges))

    lines_edges = cv2.addWeighted(color_edges, 0.8, line_image, 1, 0)
    plt.imshow(lines_edges)
    plt.show()

    st.pyplot(fig2)
    time.sleep(0.2)

st.title('Detecting Lanes | Visualizer')
st.header('Gain intuition by changing parameters into several images of lanes.')


row1_1, row1_2 = st.columns((2,2))

with row1_1, _lock:
    low_threshold = st.slider("l_threshold", 0, 300, 50)
    high_threshold = st.slider("h_threshold", 0, 300, 150)
    threshold = st.slider("threshold", 0, 300, 15)

with row1_2, _lock:
    min_line_length = st.slider("min_line_length", 0, 300, 20)
    rho = st.slider("rho", 1, 10, 2)
    max_line_gap = st.slider("max_line_gap", 0, 200, 20)

row2_1, row2_2, row2_3, row2_4 = st.columns((1,1,1,1))

with row2_1, _lock:
    st.write("**1 - Original - Solid Yellow Left**" )
    image = 'solidYellowLeft.jpeg'
    show_image(image)

with row2_2, _lock:
    st.write("**1 - Detected Lanes - Solid Yellow Left**")
    image = 'solidYellowLeft.jpeg'
    detect_lanes(image, low_threshold, high_threshold, threshold, min_line_length, rho, max_line_gap)

with row2_3, _lock:
    st.write("**2 - Original - Solid White Right**")
    image = 'solidWhiteRight.jpeg'
    show_image(image)

with row2_4, _lock:
    st.write("**2 - Detected Lanes - Solid White Righ**")
    image = 'solidWhiteRight.jpeg'
    detect_lanes(image, low_threshold, high_threshold, threshold, min_line_length, rho, max_line_gap)

row3_1, row3_2, row3_3, row3_4 = st.columns((1,1,1,1))

with row3_1, _lock:
    st.write("**3 - Original - Solid White Curve**" )
    image = 'solidWhiteCurve.jpeg'
    show_image(image)

with row3_2, _lock:
    st.write("**3 - Detected Lanes - Solid White Curve**")
    image = 'solidWhiteCurve.jpeg'
    detect_lanes(image, low_threshold, high_threshold, threshold, min_line_length, rho, max_line_gap)

with row3_3, _lock:
    st.write("**4 - Original - Solid Yellow Curve**")
    image = 'solidYellowCurve.jpeg'
    show_image(image)

with row3_4, _lock:
    st.write("**4 - Detected Lanes - Solid Yellow Curve**")
    image = 'solidYellowCurve.jpeg'
    detect_lanes(image, low_threshold, high_threshold, threshold, min_line_length, rho, max_line_gap)

st.write("by *Jhan* | [millanjhanett@gmail.com](mailto:millanjhanett@gmail.com)")