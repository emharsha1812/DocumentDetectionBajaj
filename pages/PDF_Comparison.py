import cv2
import streamlit as st


def ComparePDF():
    st.title("PDF Comparison 🆚")
    st.subheader("Compare two PDFs and find out how much similar (dissimliar) they are")

    # Create two file uploaders
    uploaded_file_1 = st.file_uploader("Original file:", type=["pdf"])
    uploaded_file_2 = st.file_uploader("Changed file:", type=["pdf"])

    # Check if both files have been uploaded
    if uploaded_file_1 is not None and uploaded_file_2 is not None:

    # Read the contents of the uploaded files
        file_1_content = uploaded_file_1.read()
        file_2_content = uploaded_file_2.read()

    # Compare the two files
        diff = difflib.unified_diff(file_1_content.splitlines(), file_2_content.splitlines(), fromfile="Original file", tofile="Changed file")

    # Display the diff to the user
        st.markdown("`diff\n" + "\n".join(diff) + "\n`")

    else:

        # Display a message to the user if either file has not been uploaded
        if uploaded_file_1 is None:
            st.warning("Please upload the original file.")
        if uploaded_file_2 is None:
            st.warning("Please upload the changed file.")



if __name__ == "__main__":
    ComparePDF()