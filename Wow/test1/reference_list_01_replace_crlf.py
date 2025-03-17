from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent


def replace_crlf(input_path, output_path):
    input_path = BASE_DIR.joinpath(input_path)
    output_path = BASE_DIR.joinpath(output_path)
    with open(input_path, 'rb') as f_in:
        content = f_in.read()

    # 只替换CRLF（\r\n）为LF（\n），不影响单独的LF
    content = content.replace(b'\r\n', b'')

    with open(output_path, 'wb') as f_out:
        f_out.write(content)


replace_crlf('Spell.11.1.0.59324.csv', 'Spell.11.1.0.59324_lfonly.csv')
replace_crlf('Spell_zhCN.11.1.0.59324.csv', 'Spell_zhCN.11.1.0.59324_lfonly.csv')
