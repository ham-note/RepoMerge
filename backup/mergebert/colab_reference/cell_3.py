a_code = "def hello(): print('A')"
b_code = "def hello(): print('B')"
o_code = "def hello(): pass"

result = resolve_conflict(a_code, b_code, o_code)
print(result)