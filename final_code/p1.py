import xlsxwriter

workbook = xlsxwriter.Workbook("data.xlsx")
worksheet = workbook.add_worksheet()
i = 0
try:
    while(True):
        worksheet.write(0, 0, i)
        i += 1

except KeyboardInterrupt:
    workbook.close()
