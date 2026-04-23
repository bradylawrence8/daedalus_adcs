import openpyxl

while(True):
    i = input("enter anything: ")
    workbook = openpyxl.load_workbook("data.xlsx")
    sheet = workbook['Sheet1']
    val = sheet['A1'].value
    print(val)
    workbook.close()
