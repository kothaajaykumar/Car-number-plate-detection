import pickle
from flask import Flask,request,app,jsonify,url_for,render_template,redirect
import numpy as np
import pandas as pd
from flask import flash
import cv2 as cv
import os
from matplotlib import pyplot as plt
import numpy as np
import imutils
import easyocr

from flask import session

UPLOAD_FOLDER = './uploads'
app=Flask(__name__, template_folder='./templates', static_folder='./static')

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['SESSION_TYPE'] = 'filesystem'
app.config['SECRET_KEY'] = 'mlappisrunninghere'


# reg_model=pickle.load(open("diabities_log_reg.pkl",'rb'))
# scaler=pickle.load(open("Scaler.pkl",'rb'))


@app.route("/",methods=['GET', 'POST'])
def index():
    data={"text":"------"}
    if request.method == "POST":
        try:
            file = request.files['image'].read()
            file_bytes = np.fromstring(file, np.uint8)
            img = cv.imdecode(file_bytes, cv.IMREAD_UNCHANGED)
            # print(img)
            gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            plt.imshow(cv.cvtColor(gray, cv.COLOR_BGR2RGB))
            # print("Hello")
            bfilter = cv.bilateralFilter(gray, 11, 17, 17) #Noise reduction
            edged = cv.Canny(bfilter, 30, 200) #Edge detection
            plt.imshow(cv.cvtColor(edged, cv.COLOR_BGR2RGB))
            keypoints = cv.findContours(edged.copy(), cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
            contours = imutils.grab_contours(keypoints)
            contours = imutils.grab_contours(keypoints)
            contours = sorted(contours, key=cv.contourArea, reverse=True)[:10]
            location = None

            for contour in contours:
                approx = cv.approxPolyDP(contour, 10, True)
                if len(approx) == 4:
                    location = approx
                    break
                    
            mask = np.zeros(gray.shape, np.uint8)
            new_image = cv.drawContours(mask, [location], 0,255, -1)
            new_image = cv.bitwise_and(img, img, mask=mask)

            (x,y) = np.where(mask==255)
            (x1, y1) = (np.min(x), np.min(y))
            (x2, y2) = (np.max(x), np.max(y))
            cropped_image = gray[x1:x2+1, y1:y2+1]

            reader = easyocr.Reader(['en'])
            result = reader.readtext(cropped_image)

            text = result[0][-2]
            font = cv.FONT_HERSHEY_SIMPLEX
            data={"text":text}
            render_template("index.html",data=data)
        except:
            return render_template("Error_page.html")
    return render_template("index.html",data=data)


if(__name__=="__main__"):
    app.run(debug=True)