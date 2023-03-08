import time
import matplotlib.pyplot as plt
import numpy as np
import string
from functools import reduce
import hashlib as hl


def function_timer(func, **kwargs) -> float:
    """Returns the execution time of the passed function and its respective arguments."""

    start_time = time.time()
    func(**kwargs)
    end_time = time.time()

    return end_time - start_time


def visualize_data(
    x_values: list,
    y_values: list,
    title: str = "",
    x_label: str = "",
    y_label: str = "",
    grid: bool = True,
) -> None:
    """Returns a graph with the lines of each function."""

    plt.plot(x_values, np.log2(y_values), "ro")
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.grid(grid)
    plt.show()


def gcd(x: int, y: int) -> int:
    """Returns the greatest common divisor of x and y."""

    if y == 0:
        return abs(x)
    else:
        return abs(gcd(y, x % y))


def coprimes(x) -> list[int]:
    """Return a list of numbers coprime to x."""

    coprimes = [num for num in range(1, x + 1) if gcd(num, x) == 1]

    return coprimes


def totient(x) -> int:
    """Returns the totient of x."""

    coprimes = [num for num in range(1, x + 1) if gcd(num, x) == 1]
    totient = len(coprimes)

    return totient


def egcd(a: int, b: int) -> int:
    """Returns the Extended Euclidean Algorithm of a and b."""
    if a == 0:
        return (b, 0, 1)
    else:
        g, y, x = egcd(b % a, a)
        return (g, x - (b // a) * y, y)


def modinv(a: int, m: int) -> int:
    """Returns the modular inverse of a and m."""

# Alternative way using Python 3.8+
# y = pow(x, -1, p)
    g, x, y = egcd(a, m)
    if g != 1:
        raise Exception("modular inverse does not exist")
    else:
        return x % m


def square_roots(n: int, mod: int) -> list:
    """Returns a list of roots of n modulo mod."""

    roots = [x for x in range(1, mod) if ((x * x) % mod == n % mod)]

    return roots


def sqrt_prime(n: int, mod: int) -> int:
    """Returns one sqrt of n if mod is a prime number."""

    e = (mod + 1) // 2

    return pow(n, e, mod)


def prime_factors(n: int) -> list:
    """Returns a list of prime factors of n."""

    i = 2
    factors = []
    while i * i <= n:
        if n % i:
            i += 1
        else:
            n //= i
            factors.append(i)
    if n > 1:
        factors.append(n)

    return factors



def chinese_remainder(m: int, a: int) -> int:
    """Returns the CRT of a and modulus m."""

    sum = 0
    prod = reduce(lambda a, b: a * b, m)
    for n_i, a_i in zip(m, a):
        p = prod // n_i
        sum += a_i * mul_inv(p, n_i) * p

    return sum % prod


def mul_inv(a: int, b: int) -> int:
    """Returns the multiplicative inverse of a and b."""

    b0 = b
    x0, x1 = 0, 1
    if b == 1:
        return 1
    while a > 1:
        q = a // b
        a, b = b, a % b
        x0, x1 = x1 - q * x0, x0
    if x1 < 0:
        x1 += b0
    return x1


def crt(n: int, mod_values: list) -> None:
    """Prints the factors of n for each mod in the list mod_values."""

    # ! Some functions are to be implemented in SageMath (sqrt, mod)
    for idx, m in enumerate(mod_values):
        sqrts = list(map(int, mod(n, m).sqrt(all=True)))
        j = len(sqrts)
        factors = []
        for i in range(j):
            fact = gcd(sqrts[i] + sqrts[i + j - 1], m)
            if fact != m and fact != 1:
                factors.append(fact)
            j -= 1
        print(f"N{idx} Factors: {factors}")


def encode_message(message: str) -> int:
    """
    Returns encoded decimal value of ascii plaintext message.
    Encoding turns all lowercase letters to uppercase for simplicity.
    """

    my_encoded = 0
    for char in message:
        my_encoded = 100 * my_encoded + ord(char.upper())

    return my_encoded


def decode_message(message: int) -> str:
    """
    Returns decoded ascii plaintext message from its encoded decimal value.
    Encoding returns uppercase letters for simplicity.
    """

    decoded_message = ""
    while message > 0:
        decoded_message = chr(message % 100) + decoded_message
        message = message // 100

    return decoded_message


def hash_message(message: str) -> str:
    """Returns a hexdecimal hash from a string."""

    hash = hl.sha256()
    hash.update(message.encode("UTF-8"))

    return hash.hexdigest()


def rsa_encrypt(plaintext: int, public_key: tuple) -> int:
    """Returns encrypted ciphertext using the RSA algorithm."""

    ciphertext = pow(plaintext, public_key[0], public_key[1])

    return ciphertext


def rsa_decrypt(ciphertext: int, private_key: tuple) -> int:
    """Returns decrypted plaintext using RSA algorithm."""

    plaintext = pow(ciphertext, private_key[0], private_key[1])

    return plaintext


def rsa_signature(hash: int, private_key: tuple) -> int:
    """Returns RSA Digital Signature"""

    signature = pow(hash, private_key[0], private_key[1])

    return signature


def generator(p: int) -> int:
    """Returns the smallest primitive root of a prime p."""

    coprimes = {n for n in range(1, p) if gcd(n, p) == 1}
    for c in coprimes:
        residues = {pow(c, n, p) for n in range(1, p)}
        if residues == coprimes:
            return c


def discrete_log(base: int, result: int, m: int) -> int:
    """Returns the discrete log k for given base, result, and m."""

    # ! Some functions are to be implemented in SageMath (sqrt)
    n = int(sqrt(m) + 1)
    value = [0] * m
    # Store all values of base^(n*i) of LHS
    for i in range(n, 0, -1):
        value[pow(base, i * n, m)] = i
    for j in range(n):
        # Calculate (base ^ j) * result and check for collision
        cur = (pow(base, j, m) * result) % m
        # If collision occurs i.e., LHS = RHS
        if value[cur]:
            k = value[cur] * n - j
            # Check whether ans lies below m or not
            if k < m:
                return k

    return -1


def all_points(a: int, b: int, m: int):
    """Returns the set of points on a non-singular elliptic curve."""

    # ! Some functions are to be implemented in SageMath (is_square, sqrt, mod)
    Z = {i for i in range(m)}
    points = set()
    for x in Z:
        residue = pow(pow(x, 3) + (a * x) + b, 1, m)
        if is_square(residue):
            y = int(mod(-1 * sqrt(residue), m))
            if (x, y) in points:
                points.add(0)
            else:
                points.add((x, y))
            if (x, sqrt(residue)) in points:
                points.add(0)
            else:
                points.add((x, sqrt(residue)))

    return points


def add_points(p: tuple, q: tuple, a: int, b: int, m: int):
    """Returns the sum of two points on an non-singular elliptic curve."""

    if tuple(map(abs, p)) == tuple(map(abs, q)):
        lam = pow((3 * pow(p[0], 2) + a) * pow(2 * p[1], -1, m), 1, m)
    else:
        lam = pow((q[1] - p[1]) * pow(q[0] - p[0], -1, m), 1, m)
    x_3 = pow(lam ** 2 - p[0] - q[0], 1, m)
    y_3 = pow(lam * (p[0] - x_3) - p[1], 1, m)
    r = (x_3, y_3)

    return r


def scalar_multiplication(n: int, P: tuple, a: int, b: int, m: int) -> tuple:
    """Returns the scalar multiplication of a point on an non-singular elliptic curve."""
    p, q = P, P
    for _ in range(1, n):
        p = add_points(p, q, a, b, m)

    return p


def check_point(x: int, y: int, a: int, b: int, m: int) -> bool:
    """Checks if a point is on a given non-singular elliptic curve."""

    if (pow(y, 2, m) == pow(pow(x, 3) + (a * x) + b, 1, m)) and (
        4 * pow(a, 3) + 27 * pow(b, 2) != 0
    ):
        return True
    return False


def caesar_cipher(message: str, shift_value: int) -> str:
    """
    Returns encoded message using Caesar Cipher.
    Input is case sensitive and eliminates foreign characters.
    """

    shift = shift_value % 27
    alphabet = string.ascii_uppercase + " "
    foreign_chars = string.digits + string.punctuation
    shifted_alphabet = alphabet[shift:] + alphabet[:shift]
    translation_table = str.maketrans(alphabet, shifted_alphabet)
    translated_message = message.upper().translate(translation_table)
    ciphertext = translated_message.translate(str.maketrans("", "", foreign_chars))

    return ciphertext


def substitution_cipher(message: str, key: str, mode: int) -> str:
    """
    Returns encoded/decoded message using randomized matrix Caesar Cipher.
    Mode takes 1 or -1 for encryption or decryption respectively. 
    """

    alphabet = string.ascii_uppercase + " "
    if mode == 1:
        translation_table = str.maketrans(alphabet, key)
    elif mode == -1:
        translation_table = str.maketrans(key, alphabet)
    else:
        raise ValueError
    alphanum_message = "".join(e for e in message if e in list(alphabet))
    ciphertext = alphanum_message.upper().translate(translation_table)

    return ciphertext


def vigenere_cipher(message: str, key: str, mode: int) -> str:
    """
    Returns encoded/decoded message using Vigenere Cipher.
    Mode takes 1 or -1 for encryption or decryption respectively.
    """

    # | LETTER: | A | B | C | D | E | F | G | H | I | J | K | L | M | N | O | P | Q | R | S | T | U | V | W | X | Y | Z | space |  
    # |:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
    # | SUBSTITUTE: | N | D | P | O | R | space | B | X | Z | L | H | A | S | U | Q | G | K | Y | J | I | W | T | V | F | M | E | C |
    
    alphabet = string.ascii_uppercase + " "
    alphanum_key = "".join(e.upper() for e in key if e in list(alphabet))
    alphanum_message = "".join(e for e in message if e in list(alphabet))
    wrapped_key = (alphanum_key * (len(alphanum_message) // len(alphanum_key) + 1))[: len(alphanum_message)]
    enumerated_alphabet = {letter: idx for idx, letter in enumerate(alphabet)}
    translated_key = [
        enumerated_alphabet[letter]
        for letter in wrapped_key
        if letter in enumerated_alphabet
    ]
    translated_message = [
        enumerated_alphabet[letter]
        for letter in alphanum_message
        if letter in enumerated_alphabet
    ]
    if mode == 1:
        shifted_message = [
            (letter + key) % 27
            for letter, key in zip(translated_message, translated_key)
        ]
    elif mode == -1:
        shifted_message = [
            (letter - key) % 27
            for letter, key in zip(translated_message, translated_key)
        ]
    else:
        raise ValueError
    ciphertext = "".join(
        list(enumerated_alphabet.keys())[list(enumerated_alphabet.values()).index(num)]
        for num in shifted_message
        if num in enumerated_alphabet.values()
    )

    return ciphertext


def one_time_pad(message: str, key: str, partial_encoding: bool = True) -> str:
    """
    Returns encoded/decoded message using One Time Pad Cipher.
    If the partial_encoding is True (default): a message longer than the key will result in partial encoding/decoding.
    If the partial_encoding is False: a message longer than the key will result in the key wrapping around.
    This uses a shifted alphabet per letter of regular alphabet to lookup encodings.
    For reference: https://en.wikipedia.org/wiki/One-time_pad#/media/File:NSA_DIANA_one_time_pad.tiff
    """
    
    alphabet = string.ascii_uppercase
    alphanum_message = "".join(e.upper() for e in message if e in alphabet)
    alphanum_key = "".join(e.upper() for e in key if e in alphabet)
    wrapped_key = (alphanum_key * (len(alphanum_message) // len(alphanum_key) + 1))[: len(alphanum_message)]
    shifted_alphabets = sorted([''.join(chr((ord(char) - 65 - e) % 26 + 65) for char in alphabet[::-1]) for e in range(1, 27)], reverse=True)
    lookup_table = dict(zip(alphabet, shifted_alphabets))
    if partial_encoding:
        encoding = zip(alphanum_message, alphanum_key)
    else:
        encoding = zip(alphanum_message, wrapped_key)
    ciphertext = "".join([lookup_table[key][alphabet.index(letter)] for letter, key in encoding])

    return ciphertext



