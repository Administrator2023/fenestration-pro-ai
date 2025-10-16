import streamlit as st

st.set_page_config(page_title="Test App")
st.title("Test BQE Integration")

# Test the problematic structure
if st.button("Test"):
    if True:
        if False:
            st.write("Path 1")
        else:
            with st.spinner("Testing..."):
                try:
                    st.write("In try block")
                    response = None
                except Exception as e:
                    st.error(f"Error: {e}")
    else:
        st.write("Path 2")

st.success("✅ If you see this, the syntax is valid!")