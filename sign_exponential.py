from pathlib import Path
import re
import shutil


ORIGINAL_DIR = Path("original")
OUTPUT_DIR = Path("mnist_sign_exponential")

FILES = {
    "w3": "w3.h",
    "w6": "w6.h",
    "b3": "b3.h",
    "b6": "b6.h",
}

EXPECTED_SIZE = {
    "w3": 100352,   
    "w6": 1280,    
    "b3": 128,
    "b6": 10,
}


def extract_array(header_path, array_name):

    text = header_path.read_text()

    pattern = (
        rf'\b{re.escape(array_name)}'
        rf'\s*\[\s*\d+\s*\]'
        rf'\s*=\s*\{{'
    )

    match = re.search(pattern, text)

    if match is None:
        raise RuntimeError(
            f"Could not find synthesis array '{array_name}' "
            f"in {header_path}"
        )

    start = match.end()

    # Find closing };
    end = text.find("};", start)

    if end == -1:
        raise RuntimeError(
            f"Could not find end of array '{array_name}' "
            f"in {header_path}"
        )

    body = text[start:end]

    number_pattern = r"""
        [-+]?
        (?:
            (?:\d+\.\d*)
            |
            (?:\.\d+)
            |
            (?:\d+)
        )
        (?:[eE][-+]?\d+)?
    """

    values = [
        float(x)
        for x in re.findall(number_pattern, body, re.VERBOSE)
    ]

    expected = EXPECTED_SIZE[array_name]

    if len(values) != expected:
        raise RuntimeError(
            f"{array_name}: expected {expected} values, "
            f"but found {len(values)}"
        )

    return values


ALLOWED_EXPONENTS = [-3, -2, -1, 0, 1, 2, 3, 4]

POT_VALUES = {
    e: 2.0 ** e
    for e in ALLOWED_EXPONENTS
}


def encode_pot(value, tolerance=1e-6):

    if abs(value) < tolerance:
        return 0, 1, 0

    sign = 1 if value > 0 else 0
    magnitude = abs(value)

    best_exp = min(
        ALLOWED_EXPONENTS,
        key=lambda e: abs(POT_VALUES[e] - magnitude)
    )

    error = abs(POT_VALUES[best_exp] - magnitude)

    if error > tolerance:
        raise RuntimeError(
            f"Non-PoT weight detected: {value:.12f}, "
            f"nearest PoT = {POT_VALUES[best_exp]}, "
            f"error = {error}"
        )

    return 1, sign, best_exp


def write_pot_weight_header(
    output_path,
    variable_name,
    values,
    cpp_type,
):
    guard = output_path.stem.upper() + "_H_"

    with output_path.open("w") as f:

        f.write(f"#ifndef {guard}\n")
        f.write(f"#define {guard}\n\n")

        f.write('#include "../defines.h"\n\n')

        f.write(
            f"{cpp_type} {variable_name}[{len(values)}] = {{\n"
        )

        for i, value in enumerate(values):

            nonzero, positive, exponent = encode_pot(value)

            f.write(
                f"    {{{nonzero}, {positive}, {exponent}}}"
            )

            if i != len(values) - 1:
                f.write(",")

            if (i + 1) % 8 == 0:
                f.write("\n")
            else:
                f.write(" ")

        f.write("\n};\n\n")
        f.write(f"#endif // {guard}\n")


def write_bias_header(
    output_path,
    variable_name,
    values,
    cpp_type,
):
    guard = output_path.stem.upper() + "_H_"

    with output_path.open("w") as f:

        f.write(f"#ifndef {guard}\n")
        f.write(f"#define {guard}\n\n")

        f.write('#include "../defines.h"\n\n')

        f.write(
            f"{cpp_type} {variable_name}[{len(values)}] = {{\n"
        )

        for i, value in enumerate(values):

            f.write(f"    {value:.10f}")

            if i != len(values) - 1:
                f.write(",")

            if (i + 1) % 4 == 0:
                f.write("\n")
            else:
                f.write(" ")

        f.write("\n};\n\n")
        f.write(f"#endif // {guard}\n")

def find_firmware_dir():
    candidates = [
        OUTPUT_DIR / "firmware",
        OUTPUT_DIR / "firmware" / "weights",
    ]

    for p in candidates:
        if p.exists():
            return p

    raise RuntimeError(
        f"Could not find firmware directory in {OUTPUT_DIR}"
    )


def patch_defines(defines_path):

    text = defines_path.read_text()

   
    text = re.sub(
        r'typedef ap_fixed<16,6> dense_weight_t;\s*\n',
        '',
        text
    )

    text = re.sub(
        r'typedef ap_fixed<16,6> dense_1_weight_t;\s*\n',
        '',
        text
    )

    # Avoid duplicate struct
    if "struct dense_weight_t" not in text:

        marker = "typedef ap_fixed<16,6> input_t;"

        struct_code = """

struct dense_weight_t {
    ap_uint<1> nonzero;
    ap_uint<1> positive;
    ap_int<4> exponent;
};

typedef dense_weight_t dense_1_weight_t;

"""

        if marker in text:
            text = text.replace(
                marker,
                struct_code + marker,
                1
            )
        else:
            text = struct_code + text

    defines_path.write_text(text)


def patch_parameters(parameters_path):

    text = parameters_path.read_text()

    if '#include "nnet_utils/pot_mult.h"' not in text:

        marker = '#include "nnet_utils/nnet_dense.h"'

        text = text.replace(
            marker,
            marker + '\n#include "nnet_utils/pot_mult.h"',
            1
        )

    text = re.sub(
        r'using product = nnet::product::mult<x_T, y_T>;',
        r'using product = nnet::product::pot_mult<x_T, y_T>;',
        text
    )

    parameters_path.write_text(text)


def patch_myproject(myproject_path):

    text = myproject_path.read_text()

   
    pattern = (
        r'#ifndef __SYNTHESIS__'
        r'.*?'
        r'#endif'
    )

    matches = list(
        re.finditer(pattern, text, re.DOTALL)
    )

    for match in reversed(matches):

        block = match.group(0)

        if "load_weights_from_txt" in block:
            text = (
                text[:match.start()]
                + text[match.end():]
            )
            break

    myproject_path.write_text(text)



def write_pot_mult(path):

    text = r'''#ifndef POT_MULT_H_
#define POT_MULT_H_

#include "ap_fixed.h"
#include "ap_int.h"

namespace nnet {
namespace product {

template <class x_T, class w_T>
class pot_mult {
public:

    typedef ap_fixed<40,20> result_t;

    static result_t product(x_T &a, w_T &w) {

        #pragma HLS INLINE

        if (!w.nonzero)
            return result_t(0);

        result_t x = static_cast<result_t>(a);
        result_t y;

        switch ((int)w.exponent) {

            case -3:
                y = x >> 3;
                break;

            case -2:
                y = x >> 2;
                break;

            case -1:
                y = x >> 1;
                break;

            case 0:
                y = x;
                break;

            default:
                y = 0;
                break;
        }

        if (!w.positive)
            y = -y;

        return y;
    }
};

} // namespace product
} // namespace nnet

#endif
'''

    path.write_text(text)


def main():

    for name, filename in FILES.items():

        path = ORIGINAL_DIR / filename

        if not path.exists():
            raise FileNotFoundError(
                f"Missing: {path}"
            )


    if OUTPUT_DIR.exists():
        print(f"\nRemoving old output: {OUTPUT_DIR}")
        shutil.rmtree(OUTPUT_DIR)

    print(f"\nCopying original project -> {OUTPUT_DIR}")

    source_project = Path("mnist_hls_resource_approx_pot_test")

    if source_project.exists():

        shutil.copytree(
            source_project,
            OUTPUT_DIR
        )

    else:

        raise RuntimeError(
            "\nCould not find original HLS project:\n"
            "  mnist_hls_resource_approx_pot_test\n\n"
            "Put the untouched hls4ml project next to this script."
        )

    firmware = OUTPUT_DIR / "firmware"
    weights_dir = firmware / "weights"

    weights_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    print("\nReading original hls4ml arrays...")

    w3 = extract_array(
        ORIGINAL_DIR / "w3.h",
        "w3"
    )

    w6 = extract_array(
        ORIGINAL_DIR / "w6.h",
        "w6"
    )

    b3 = extract_array(
        ORIGINAL_DIR / "b3.h",
        "b3"
    )

    b6 = extract_array(
        ORIGINAL_DIR / "b6.h",
        "b6"
    )


    def stats(name, values):

        zeros = sum(
            abs(x) < 1e-6
            for x in values
        )

        nonzeros = len(values) - zeros

        print(
            f"{name}: "
            f"size={len(values)}, "
            f"zeros={zeros}, "
            f"nonzeros={nonzeros}, "
            f"min={min(values):.6f}, "
            f"max={max(values):.6f}"
        )

    print()

    stats("w3", w3)
    stats("w6", w6)
    stats("b3", b3)
    stats("b6", b6)


    print("\nChecking PoT weights...")

    for name, values in [
        ("w3", w3),
        ("w6", w6),
    ]:

        bad = []

        for i, value in enumerate(values):

            if abs(value) < 1e-6:
                continue

            try:
                encode_pot(value)
            except RuntimeError:
                bad.append((i, value))

        if bad:

            print(
                f"\nERROR: {name} contains "
                f"{len(bad)} non-PoT values."
            )

            for i, value in bad[:10]:
                print(
                    f"  index={i}, value={value}"
                )

            raise RuntimeError(
                f"{name} is not a pure PoT array."
            )

        print(
            f"{name}: all nonzero weights are valid PoT."
        )

    print("\nGenerating PoT weight headers...")

    write_pot_weight_header(
        weights_dir / "w3.h",
        "w3",
        w3,
        "dense_weight_t"
    )

    write_pot_weight_header(
        weights_dir / "w6.h",
        "w6",
        w6,
        "dense_1_weight_t"
    )

    print("Generating bias headers...")

    write_bias_header(
        weights_dir / "b3.h",
        "b3",
        b3,
        "dense_bias_t"
    )

    write_bias_header(
        weights_dir / "b6.h",
        "b6",
        b6,
        "dense_1_bias_t"
    )


    defines_path = firmware / "defines.h"

    print("Patching defines.h...")

    patch_defines(defines_path)

    

    pot_mult_path = firmware / "nnet_utils" / "pot_mult.h"

    pot_mult_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    print("Writing pot_mult.h...")

    write_pot_mult(pot_mult_path)


    parameters_path = firmware / "parameters.h"

    print("Patching parameters.h...")

    patch_parameters(parameters_path)


    myproject_path = firmware / "myproject.cpp"

    print("Patching myproject.cpp...")

    patch_myproject(myproject_path)


    print("\n" + "*" * 50)
    print("DONE")
    print("*" * 50)

    print("\nGenerated:")

    print(f"  {weights_dir / 'w3.h'}")
    print(f"  {weights_dir / 'w6.h'}")
    print(f"  {weights_dir / 'b3.h'}")
    print(f"  {weights_dir / 'b6.h'}")
    print(f"  {pot_mult_path}")


if __name__ == "__main__":
    main()