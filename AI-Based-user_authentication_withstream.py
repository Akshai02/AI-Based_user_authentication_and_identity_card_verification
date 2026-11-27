
# ============================================================
# AI-BASED USER AUTHENTICATION SYSTEM
# Face Recognition + ID Card Verification
# ============================================================

# STEP 1: IMPORT REQUIRED LIBRARIES

import streamlit as st
import cv2
import numpy as np
import os
import face_recognition as fr
import pytesseract
import re

from datetime import datetime


# ============================================================
# STEP 2: STREAMLIT PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AI User Authentication",
    page_icon="🔐",
    layout="centered"
)

st.title("🔐 AI-Based User Authentication System")

st.subheader(
    "Face Recognition and Identity Card Verification"
)

st.write(
    "Authenticate users using face recognition "
    "and identity card verification."
)


# ============================================================
# STEP 3: CONFIGURE TESSERACT OCR
# ============================================================

pytesseract.pytesseract.tesseract_cmd = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)


# ============================================================
# STEP 4: REGISTERED USER DATABASE
# ============================================================

user_data = {

    "AARON LOEB": {
        "DOB": "30/03/1998"
    },

    "AKSHAI A K": {
        "DOB": "18/02/1999"
    },

    "ATHIRA ANIL": {
        "DOB": "20/03/2000"
    }
}


# ============================================================
# STEP 5: LOAD REGISTERED FACE IMAGES
# ============================================================

path = "Images"


@st.cache_resource
def load_registered_faces():

    img = []
    classnames = []

    # Read all images from Images folder
    mylist = os.listdir(path)

    for i in mylist:

        imgpath = os.path.join(path, i)

        current_img = cv2.imread(imgpath)

        # Ignore unreadable images
        if current_img is None:

            print("Unable to read:", i)
            continue

        img.append(current_img)

        # Extract name from filename
        classnames.append(
            os.path.splitext(i)[0].upper()
        )

    return img, classnames


# ============================================================
# STEP 6: GENERATE FACE ENCODINGS
# ============================================================

def face_encodings(images):

    encode_list = []

    for i in images:

        # Convert BGR to RGB
        img = cv2.cvtColor(
            i,
            cv2.COLOR_BGR2RGB
        )

        # Detect faces
        face_in_frame = fr.face_locations(img)

        # Skip images without faces
        if len(face_in_frame) == 0:

            encode_list.append(None)
            continue

        # Generate face encoding
        encode = fr.face_encodings(
            img,
            face_in_frame
        )[0]

        encode_list.append(encode)

    return encode_list


# Load registered images
try:

    img, classnames = load_registered_faces()

    encode_list_known_face = face_encodings(img)

    st.success("Registered face images loaded.")

except Exception as e:

    st.error(f"Unable to load registered faces: {e}")
    st.stop()


# Remove invalid encodings

valid_encodings = []
valid_names = []

for encoding, name in zip(
    encode_list_known_face,
    classnames
):

    if encoding is not None:

        valid_encodings.append(encoding)
        valid_names.append(name)


if len(valid_encodings) == 0:

    st.error("No valid registered face encodings found.")
    st.stop()


# ============================================================
# STEP 7: FACE RECOGNITION FUNCTION
# ============================================================

def recognize_face(uploaded_face):

    # Convert uploaded image bytes into OpenCV image
    file_bytes = np.asarray(
        bytearray(uploaded_face.getvalue()),
        dtype=np.uint8
    )

    frame = cv2.imdecode(
        file_bytes,
        cv2.IMREAD_COLOR
    )

    if frame is None:

        return None, None

    # Convert BGR to RGB
    frame1 = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    # Detect faces
    face_in_frame = fr.face_locations(frame1)

    if len(face_in_frame) == 0:

        return None, frame

    # Generate face encodings
    encode_value = fr.face_encodings(
        frame1,
        face_in_frame
    )

    recognized_name = None

    # Compare captured face with registered faces
    for enc_face, faceloc in zip(
        encode_value,
        face_in_frame
    ):

        matches = fr.compare_faces(
            valid_encodings,
            enc_face
        )

        face_dis = fr.face_distance(
            valid_encodings,
            enc_face
        )

        match_index = np.argmin(face_dis)

        if matches[match_index]:

            name = valid_names[match_index]

        else:

            name = "UNKNOWN"

        # Draw rectangle
        y1, x2, y2, x1 = faceloc

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (0, 0, 255),
            3
        )

        cv2.putText(
            frame,
            name,
            (x1, max(y1 - 10, 20)),
            cv2.FONT_HERSHEY_COMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

        if name != "UNKNOWN":

            recognized_name = name
            break

    return recognized_name, frame


# ============================================================
# STEP 8: CAPTURE FACE USING STREAMLIT CAMERA
# ============================================================

st.header("Step 1: Face Recognition")

st.write("Capture your face using the camera.")

uploaded_face = st.camera_input(
    "Capture your face"
)


# ============================================================
# STEP 9: UPLOAD ID CARD
# ============================================================

st.header("Step 2: Identity Card Verification")

id_file = st.file_uploader(
    "Upload your identity card",
    type=["jpg", "jpeg", "png"]
)


# ============================================================
# STEP 10: VERIFY IDENTITY BUTTON
# ============================================================

if st.button(
    "Verify Identity",
    type="primary",
    use_container_width=True
):

    # Check face image
    if uploaded_face is None:

        st.warning("Please capture your face.")

        st.stop()

    # Check ID card image
    if id_file is None:

        st.warning("Please upload your identity card.")

        st.stop()

    # ========================================================
    # STEP 11: PERFORM FACE RECOGNITION
    # ========================================================

    st.subheader("Face Recognition Result")

    with st.spinner("Recognizing face..."):

        recognized_name, face_result = recognize_face(
            uploaded_face
        )

    if face_result is not None:

        st.image(
            cv2.cvtColor(
                face_result,
                cv2.COLOR_BGR2RGB
            ),
            caption="Face Recognition Result",
            use_container_width=True
        )

    if recognized_name is None:

        st.error("Face not recognized. Access denied.")

        st.stop()

    st.success("Face recognized successfully!")

    st.write(
        "Recognized User:",
        recognized_name
    )

    # ========================================================
    # STEP 12: READ ID CARD IMAGE
    # ========================================================

    st.subheader("ID Card OCR Result")

    id_bytes = np.asarray(
        bytearray(id_file.getvalue()),
        dtype=np.uint8
    )

    id_img = cv2.imdecode(
        id_bytes,
        cv2.IMREAD_COLOR
    )

    if id_img is None:

        st.error("Unable to read ID card image.")

        st.stop()

    # Keep original image for OCR
    id_original = id_img.copy()

    # ========================================================
    # STEP 13: EXTRACT TEXT USING TESSERACT OCR
    # ========================================================

    with st.spinner("Extracting ID card information..."):

        text = pytesseract.image_to_string(
            id_original
        )

        data = pytesseract.image_to_data(
            id_original,
            output_type=pytesseract.Output.DICT
        )

    n = len(data["text"])

    st.write("Extracted ID Card Text:")

    st.text(text)

    # ========================================================
    # STEP 14: NAME EXTRACTION AND DATABASE MATCHING
    # ========================================================

    database_names = list(user_data.keys())

    ocr_words = []

    for i in range(n):

        word = data["text"][i].strip()

        # Remove unnecessary special characters
        word = re.sub(
            r"[^A-Za-z]",
            "",
            word
        )

        if word:

            ocr_words.append(
                (word.upper(), i)
            )

    # Initialize variables
    name_match = False

    extracted_name = None

    name_coordinates = []

    # Check all registered database names
    for database_name in database_names:

        database_name_clean = re.sub(
            r"\s+",
            "",
            database_name.upper()
        )

        # Search consecutive OCR words
        for i in range(len(ocr_words)):

            combined_name = ""

            matched_indexes = []

            for word, index in ocr_words[i:]:

                combined_name += word

                matched_indexes.append(index)

                if combined_name == database_name_clean:

                    name_match = True

                    extracted_name = database_name

                    name_coordinates = (
                        matched_indexes.copy()
                    )

                    break

                if len(combined_name) >= len(
                    database_name_clean
                ):

                    break

            if name_match:

                break

        if name_match:

            break

    # Draw rectangle around matched name
    if name_match:

        for index in name_coordinates:

            x = data["left"][index]
            y = data["top"][index]

            w = data["width"][index]
            h = data["height"][index]

            cv2.rectangle(
                id_img,
                (x, y),
                (x + w, y + h),
                (0, 255, 0),
                2
            )

        st.success(
            f"Database name found: {extracted_name}"
        )

    else:

        st.error(
            "No registered database name found in ID card."
        )

    # ========================================================
    # STEP 15: DOB EXTRACTION USING REGULAR EXPRESSION
    # ========================================================

    dob_pattern = (
        r"\b\d{2}[/.-]\d{2}[/.-]\d{4}\b"
    )

    extracted_dob = None

    dob_index = None

    # Search each OCR word
    for i in range(n):

        word = data["text"][i].strip()

        dob_found = re.findall(
            dob_pattern,
            word
        )

        if dob_found:

            extracted_dob = dob_found[0]

            dob_index = i

            break

    # Validate DOB
    if extracted_dob is not None:

        extracted_dob = (
            extracted_dob.replace("-", "/")
            .replace(".", "/")
        )

        try:

            extracted_dob = datetime.strptime(
                extracted_dob,
                "%d/%m/%Y"
            ).strftime("%d/%m/%Y")

        except ValueError:

            extracted_dob = None

    # Draw rectangle around DOB
    if extracted_dob is not None and dob_index is not None:

        x = data["left"][dob_index]
        y = data["top"][dob_index]

        w = data["width"][dob_index]
        h = data["height"][dob_index]

        cv2.rectangle(
            id_img,
            (x, y),
            (x + w, y + h),
            (0, 0, 255),
            3
        )

        st.write(
            "Extracted DOB:",
            extracted_dob
        )

    else:

        st.warning("DOB not found or invalid.")

    # ========================================================
    # STEP 16: DATABASE DOB COMPARISON
    # ========================================================

    if name_match:

        database_dob = user_data[
            extracted_name
        ]["DOB"]

    else:

        database_dob = None

    dob_match = (
        extracted_dob is not None
        and database_dob is not None
        and extracted_dob == database_dob
    )

    # ========================================================
    # STEP 17: DISPLAY PROCESSED ID CARD
    # ========================================================

    st.subheader("Processed Identity Card")

    st.image(
        cv2.cvtColor(
            id_img,
            cv2.COLOR_BGR2RGB
        ),
        caption="OCR Bounding Boxes",
        use_container_width=True
    )
    # ========================================================
    # FACE NAME AND ID CARD NAME COMPARISON
    # ========================================================

    face_id_match = (
        recognized_name.upper() == extracted_name.upper()
        if name_match and extracted_name is not None
        else False
    )

    # ========================================================
    # STEP 18: FINAL AUTHENTICATION RESULT
    # ========================================================

    st.header("Step 3: Authentication Result")

    col1, col2 = st.columns(2)

    with col1:
        if recognized_name is not None:
            st.success("Face Recognition: SUCCESS")
        else:
            st.error("Face Recognition: FAILED")

    with col2:
        if name_match:
            st.success("Name Verification: SUCCESS")
        else:
            st.error("Name Verification: FAILED")

    if face_id_match:
        st.success("Face-ID Name Matching: SUCCESS")
    else:
        st.error("Face-ID Name Matching: FAILED")

    if dob_match:
        st.success("DOB Verification: SUCCESS")
    else:
        st.error("DOB Verification: FAILED")

    # All four conditions must pass
    if (
        recognized_name is not None
        and name_match
        and dob_match
        and face_id_match
    ):
        st.success(
            "🎉 USER AUTHENTICATED — ACCESS GRANTED!"
        )
    else:
        st.error(
            "❌ AUTHENTICATION FAILED — ACCESS DENIED!"
        )

    # ========================================================
    # STEP 18: FINAL AUTHENTICATION RESULT
    # ========================================================

    # st.header("Step 3: Authentication Result")

    # col1, col2 = st.columns(2)

    # with col1:

    #     if recognized_name is not None:

    #         st.success("Face Recognition: SUCCESS")

    #     else:

    #         st.error("Face Recognition: FAILED")

    # with col2:

    #     if name_match:

    #         st.success("Name Verification: SUCCESS")

    #     else:

    #         st.error("Name Verification: FAILED")
    #     if face_id_match:
    #         st.success("Face-ID Name Matching: SUCCESS")
    #     else:
    #         st.error("Face-ID Name Matching: FAILED")

    # if dob_match:

    #     st.success("DOB Verification: SUCCESS")

    # else:

    #     st.error("DOB Verification: FAILED")

    # # All three checks must pass
    # if (
    #     recognized_name is not None
    #     and name_match
    #     and dob_match
    #     and face_id_match
    # ):

    #     st.success(
    #         "🎉 USER AUTHENTICATED — ACCESS GRANTED!"
    #     )

    # else:

    #     st.error(
    #         "❌ AUTHENTICATION FAILED — ACCESS DENIED!"
    #     )
