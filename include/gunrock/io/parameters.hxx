#include <cxxopts.hpp>
#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <numeric>
#include <random>
#include <sstream>
#include <string>
#include <vector>

namespace gunrock {
namespace io {
namespace cli {

struct parameters_t {
  std::string filename;
  std::string source_string = "";
  std::string json_dir = ".";
  std::string json_file = "";
  std::string tag_string = "";
  int num_runs = 1;
  int source_count = -1;
  cxxopts::Options options;
  bool export_metrics = false;
  bool validate = false;
  bool binary = false;

  /**
   * @brief Construct a new parameters object and parse command line arguments.
   *
   * @param argc Number of command line arguments.
   * @param argv Command line arguments.
   */
  parameters_t(int argc, char** argv, std::string algorithm)
      : options(argv[0], algorithm + " example") {
    // Add command line options
    options.add_options()("help", "Print help")  // help
        ("export_metrics",
         "export performance analysis metrics")  // performance evaluation
        ("m,market", "Matrix file", cxxopts::value<std::string>())  // mtx file
        ("d,json_dir", "JSON output directory",
         cxxopts::value<std::string>())  // json output directory
        ("f,json_file", "JSON output file",
         cxxopts::value<std::string>())  // json output file
        ("t,tag", "Tags for the JSON output; comma-separated string of tags",
         cxxopts::value<std::string>());  // tags

    // Algorithms with sources
    if (algorithm == "Betweenness Centrality" ||
        algorithm == "Breadth First Search" ||
        algorithm == "Single Source Shortest Path" ||
        algorithm == "Multi Source Shortest Path" || algorithm == "DAWN") {
      options.add_options()("s,src",
                            "Source(s) (random if omitted); "
                            "comma-separated ints",
                            cxxopts::value<std::string>())  // source
          ("src-count", "Number of unique random sources to generate",
           cxxopts::value<int>())
          ("n,num_runs", "Number of runs (ignored if multiple sources passed)",
           cxxopts::value<int>());  // runs
      if (algorithm == "Breadth First Search" ||
          algorithm == "Single Source Shortest Path" ||
          algorithm == "Multi Source Shortest Path" || algorithm == "DAWN") {
        options.add_options()("validate", "CPU validation");  // validate
      }
    } else {
      options.add_options()("n,num_runs", "Number of runs",
                            cxxopts::value<int>());  // runs
    }

    // Parse command line arguments
    auto result = options.parse(argc, argv);

    if (result.count("help") || (result.count("market") == 0)) {
      std::cout << options.help({""}) << std::endl;
      std::exit(0);
    }

    if (result.count("market") == 1) {
      filename = result["market"].as<std::string>();
      if (util::is_binary_csr(filename)) {
        binary = true;
      } else if (!util::is_market(filename)) {
        std::cout << options.help({""}) << std::endl;
        std::exit(0);
      }
    } else {
      std::cout << options.help({""}) << std::endl;
      std::exit(0);
    }

    if (result.count("validate") == 1) {
      validate = true;
    }

    if (result.count("export_metrics") == 1) {
      export_metrics = true;
    }

    if (result.count("num_runs") == 1) {
      num_runs = result["num_runs"].as<int>();
    }

    if (result.count("tag") == 1) {
      tag_string = result["tag"].as<std::string>();
    }

    if (result.count("src") == 1) {
      source_string = result["src"].as<std::string>();
    }

    if (result.count("src-count") == 1) {
      source_count = result["src-count"].as<int>();
    }

    if (result.count("json_dir") == 1) {
      json_dir = result["json_dir"].as<std::string>();
    }

    if (result.count("json_file") == 1) {
      json_file = result["json_file"].as<std::string>();
    }
  }
};

void parse_source_tokens(std::string source_str,
                         std::vector<int>* source_vect,
                         int n_vertices) {
  std::replace(source_str.begin(), source_str.end(), ',', ' ');
  std::stringstream ss(source_str);
  std::string source;

  while (ss >> source) {
    char* end = nullptr;
    long parsed_source = std::strtol(source.c_str(), &end, 10);
    if (end == source.c_str() || *end != '\0') {
      std::cout << "Error: Invalid source: " << source << "\n";
      exit(1);
    }
    if (parsed_source >= 0 && parsed_source < n_vertices) {
      int source_int = static_cast<int>(parsed_source);
      source_vect->push_back(source_int);
    } else {
      std::cout << "Error: Source out of range: " << parsed_source << "\n";
      exit(1);
    }
  }
}

void parse_source_string(std::string source_str,
                         std::vector<int>* source_vect,
                         int n_vertices,
                         int n_runs) {
  if (source_str == "") {
    // Generate random starting source
    std::random_device seed;
    std::mt19937 engine(seed());
    for (int i = 0; i < n_runs; i++) {
      std::uniform_int_distribution<int> dist(0, n_vertices - 1);
      source_vect->push_back(dist(engine));
    }
  } else {
    parse_source_tokens(source_str, source_vect, n_vertices);
    if (source_vect->empty()) {
      std::cout << "Error: No valid source nodes found"
                << "\n";
      exit(1);
    }
    if (source_vect->size() == 1) {
      source_vect->insert(source_vect->end(), n_runs - 1, source_vect->at(0));
    }
  }
}

void generate_unique_random_sources(int source_count,
                                    std::vector<int>* source_vect,
                                    int n_vertices) {
  if (source_count <= 0 || source_count > n_vertices) {
    std::cout << "Error: --src-count must be between 1 and the number of graph vertices ("
              << n_vertices << "): " << source_count << "\n";
    exit(1);
  }

  std::vector<int> vertices(n_vertices);
  std::iota(vertices.begin(), vertices.end(), 0);
  std::random_device seed;
  std::mt19937 engine(seed());
  std::shuffle(vertices.begin(), vertices.end(), engine);
  source_vect->insert(source_vect->end(), vertices.begin(),
                      vertices.begin() + source_count);
}

void parse_tag_string(std::string tag_str, std::vector<std::string>* tag_vect) {
  std::stringstream ss(tag_str);
  while (ss.good()) {
    std::string tag;
    getline(ss, tag, ',');
    if (tag != "") {
      tag_vect->push_back(tag);
    }
  }
}

}  // namespace cli
}  // namespace io
}  // namespace gunrock
