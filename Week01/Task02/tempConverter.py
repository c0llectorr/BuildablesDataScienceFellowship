def convert_temperature(value, scale):
    scale = scale.upper()
    if scale == "C":
        c = value
        f = (c * 9/5) + 32
        k = c + 273.15
    elif scale == "F":
        f = value
        c = (f - 32) * 5/9
        k = c + 273.15
    elif scale == "K":
        k = value
        c = k - 273.15
        f = (c * 9/5) + 32
    else:
        return "Invalid scale. Use 'C', 'F', or 'K'."

    return f"{c:.2f}°C = {f:.2f}°F = {k:.2f}K"
    
def main():
    try:
        value = float(input("Enter the temperature value: "))
        scale = input("Enter the scale (C, F, K): ")
        result = convert_temperature(value, scale)
        print(result)
    except ValueError:
        print("Please enter a valid number for the temperature value.")

if __name__ == "__main__":
    main()