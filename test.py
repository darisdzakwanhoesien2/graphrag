from modules.data_loader import load_report, BASE_PATH

print("BASE PATH:", BASE_PATH)

docs = load_report("2_2 Thermal properties and temperature QP_pdf")

print("Number of docs:", len(docs))

if docs:
    print("First 200 chars:\n")
    print(docs[0]["text"][:200])
else:
    print("No documents loaded.")
