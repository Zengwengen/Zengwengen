import docx

# 打开Word文档
doc = docx.Document('prog.docx')

# 遍历每一个段落
for para in doc.paragraphs:
    text = para.text
    flag = False
    start = 0
    end = 0
    # 遍历每一个字符
    for i in range(len(text)):
        if text[i] == '=':
            flag = True
            start = i
        elif text[i] == '#' and flag:
            end = i
            # 删除标记到#之间的内容
            para.text = para.text[:start] + para.text[end+1:]
            # 重新设置标记
            text = para.text
            flag = False
            start = 0
            end = 0
            break

# 保存处理好的文档
doc.save('processed.docx')
