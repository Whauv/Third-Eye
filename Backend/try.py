import streamlit as st
import cv2
import numpy as np
from streamlit_webrtc import VideoTransformerBase, webrtc_streamer

class MotionDetection(VideoTransformerBase):
    def __init__(self):
        self.thres = 0.5
        self.frame_width = 640
        self.frame_height = 480
        # ... (all other variables and settings from your original code)

    def transform(self, frame):
        img = cv2.cvtColor(frame.to_ndarray(), cv2.COLOR_BGR2RGB)

        # ... (all the processing from your original code, using 'img' instead of 'Control_Frame')

        return img

def main():
    st.title("Motion Detection with Streamlit")
    st.sidebar.title("Settings")

    # ... (add any Streamlit widgets for settings here, if needed)

    webrtc_ctx = webrtc_streamer(key="motion_detection", video_transformer_factory=MotionDetection)

    if st.button("Stop"):
        webrtc_ctx.stop_all()

if __name__ == "__main__":
    main()
