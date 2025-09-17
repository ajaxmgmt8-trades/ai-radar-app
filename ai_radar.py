import streamlit as st

st.subheader("🔁 Option Contract Historic Data")
contract_id = st.text_input("Enter Contract ID (e.g. 123456)", "")

if st.button("Get Historic Data") and contract_id:
    result = get_option_contract_history(contract_id)
    if "error" in result:
        st.error(result["error"])
    else:
        st.json(result)
