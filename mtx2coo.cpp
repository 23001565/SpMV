#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include <algorithm>

// Structure to hold a matrix element
struct Element {
    int row;
    int col;
    double val;
};

bool isComment(const std::string& line) {
    return line.length() > 0 && line[0] == '%';
}

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <input.mtx> <output.coo>" << std::endl;
        return 1;
    }

    std::ifstream infile(argv[1]);
    std::ofstream outfile(argv[2]);

    if (!infile.is_open()) {
        std::cerr << "Error opening input file." << std::endl;
        return 1;
    }

    std::string line;
    bool isSymmetric = false;
    bool isPattern = false;

    // 1. Parse Header
    if (std::getline(infile, line)) {
        if (line.find("symmetric") != std::string::npos) isSymmetric = true;
        if (line.find("pattern") != std::string::npos) isPattern = true;
    }

    // 2. Skip comments
    while (std::getline(infile, line)) {
        if (!isComment(line)) break;
    }

    // 3. Read Dimensions (M, N, NNZ)
    int M, N, NNZ_declared;
    std::stringstream ss(line);
    ss >> M >> N >> NNZ_declared;

    std::vector<Element> elements;
    elements.reserve(NNZ_declared); // Pre-allocate

    // 4. Read Data
    int r, c;
    double v;
    
    // We loop through the rest of the file
    // Note: We use a loop here because some mtx files might have whitespace quirks
    while(infile >> r >> c) {
        // Handle Pattern (no value provided) vs Real (value provided)
        if (isPattern) {
            v = 1.0; // Default fill value
        } else {
            infile >> v;
        }
	
	// Remove zeros
	if (v == 0.0) continue;

        // Convert 1-based to 0-based
        elements.push_back({r - 1, c - 1, v});

        // Handle Symmetry: If (r, c) exists, add (c, r)
        // Check r != c to avoid duplicating the diagonal
        if (isSymmetric && r != c) {
            elements.push_back({c - 1, r - 1, v});
        }
    }

    // 5. Write Output
    // Format: Rows Cols Total_Effective_NNZ
    // Then lines of: r c v
    outfile << M << " " << N << " " << elements.size() << "\n";
    for (const auto& el : elements) {
        outfile << el.row << " " << el.col << " " << el.val << "\n";
    }

    std::cout << "Conversion Complete.\n";
    std::cout << "Original NNZ: " << NNZ_declared << "\n";
    std::cout << "Final NNZ (after symmetry expansion): " << elements.size() << "\n";

    infile.close();
    outfile.close();
    return 0;
}