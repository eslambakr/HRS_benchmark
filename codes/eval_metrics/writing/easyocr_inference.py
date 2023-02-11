import easyocr


reader = easyocr.Reader(['en'])  # this needs to run only once to load the model into memory
result = reader.readtext('/media/eslam/0d208863-5cdb-4a43-9794-3ca8726831b3/T2I_benchmark/data/t2i_out/sd_v1/writing_15/00003_04.png')
print(result)
