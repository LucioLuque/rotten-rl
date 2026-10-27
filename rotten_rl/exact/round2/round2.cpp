// La ronda de 2 cartas, resuelta con MCCFR (muestreo externo).
//
// Ver README.md de esta carpeta. Las posiciones van en orden de canto: la 0
// canta primero y sale en la primera baza; la n-1 canta ultima (reparte). La
// segunda baza es forzada (una carta cada uno) y la sale quien gano la primera.
//
// Comandos:
//   round2 solve --n 3 --iters 1000000 [--abstraction b|exact] [--load F] [--save F]
//   round2 eval  --n 3 --games 200000 --load F --policies avg,heur,heur
//   round2 br    --n 3 --games 2000000 --load F   (cuanto gana quien se desvia)
//   round2 dump  --n 3 --load F --save estrategia.tsv
//   round2 trace --n 3 --games 200        (para comparar con el motor de Python)

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <numeric>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

constexpr int kRanks = 13;
constexpr int kMaxPlayers = 7;
constexpr int kMaxActions = 3;

struct Args {
  std::string cmd, abstraction = "b", load, save, policies = "avg";
  int n = 3;
  long long iters = 100000, games = 100000, report_every = 0;
  uint64_t seed = 1;
};

inline int Suit(int c) { return c / kRanks; }
inline int Rank(int c) { return c % kRanks; }

// splitmix64: rapido y con buena calidad para muestrear.
struct Rng {
  uint64_t s;
  explicit Rng(uint64_t seed) : s(seed) {}
  uint64_t Next() {
    uint64_t z = (s += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
  }
  double Uniform() { return (Next() >> 11) * (1.0 / 9007199254740992.0); }
  int Below(int n) { return static_cast<int>(Next() % static_cast<uint64_t>(n)); }
};

struct Deal {
  int n;
  int trump;          // carta dada vuelta
  int ts;             // palo de triunfo
  int hand[kMaxPlayers][2];
};

void DealCards(Deal& d, int n, Rng& rng) {
  int deck[52];
  std::iota(deck, deck + 52, 0);
  for (int i = 0; i < 2 * n + 1; ++i) std::swap(deck[i], deck[i + rng.Below(52 - i)]);
  d.n = n;
  d.trump = deck[0];
  d.ts = Suit(d.trump);
  for (int p = 0; p < n; ++p) {
    d.hand[p][0] = std::min(deck[1 + 2 * p], deck[2 + 2 * p]);
    d.hand[p][1] = std::max(deck[1 + 2 * p], deck[2 + 2 * p]);
  }
}

// Lo mismo que evaluation._beats: el triunfo le gana a todo lo que no es
// triunfo; dentro del mismo palo gana la mas alta; otro palo no gana.
bool Beats(int c, int best, int ts) {
  if (Suit(c) == ts && Suit(best) != ts) return true;
  if (Suit(best) == ts && Suit(c) != ts) return false;
  if (Suit(c) == Suit(best)) return Rank(c) > Rank(best);
  return false;
}

// Estado de la ronda durante el recorrido.
struct St {
  int bids[kMaxPlayers];
  int nb = 0;                  // cantos hechos
  int play1[kMaxPlayers];      // primera baza, en orden de posicion (la 0 sale)
  int np = 0;
};

int LegalBids(int n, int k, const St& st, int* out) {
  int prev = 0;
  for (int j = 0; j < k; ++j) prev += st.bids[j];
  int m = 0;
  for (int b = 0; b <= 2; ++b)
    if (!(k == n - 1 && prev + b == 2)) out[m++] = b;   // la suma no puede dar 2
  return m;
}

// Cartas jugables en la primera baza (en orden de id, como pyspiel).
int LegalPlays(const Deal& d, int k, const St& st, int* out) {
  const int* h = d.hand[k];
  if (k == 0) { out[0] = h[0]; out[1] = h[1]; return 2; }
  int lead = Suit(st.play1[0]);
  int m = 0;
  for (int i = 0; i < 2; ++i) if (Suit(h[i]) == lead) out[m++] = h[i];
  if (m) return m;
  for (int i = 0; i < 2; ++i) if (Suit(h[i]) == d.ts) out[m++] = h[i];   // obligacion de fallar
  if (m) return m;
  out[0] = h[0]; out[1] = h[1];
  return 2;
}

int TrickWinner(const int* cards, const int* order, int n, int ts) {
  int w = order[0];
  for (int i = 1; i < n; ++i)
    if (Beats(cards[order[i]], cards[w], ts)) w = order[i];
  return w;
}

// Bazas de cada posicion al terminar la ronda.
void Tricks(const Deal& d, const St& st, int* tricks) {
  int n = d.n, order[kMaxPlayers] = {}, rem[kMaxPlayers];
  for (int i = 0; i < n; ++i) order[i] = i;
  int w1 = TrickWinner(st.play1, order, n, d.ts);
  for (int p = 0; p < n; ++p) rem[p] = d.hand[p][0] == st.play1[p] ? d.hand[p][1] : d.hand[p][0];
  for (int i = 0; i < n; ++i) order[i] = (w1 + i) % n;
  int w2 = TrickWinner(rem, order, n, d.ts);
  for (int p = 0; p < n; ++p) tricks[p] = (p == w1) + (p == w2);
}

int Score(int bid, int tricks) { return tricks == bid ? 10 + 2 * tricks : tricks; }

// Reward del agente (reward_scheme=relative): puntaje menos el promedio de los rivales.
void Utilities(const Deal& d, const St& st, double* u) {
  int n = d.n, tricks[kMaxPlayers];
  Tricks(d, st, tricks);
  double s[kMaxPlayers], total = 0;
  for (int p = 0; p < n; ++p) total += s[p] = Score(st.bids[p], tricks[p]);
  for (int p = 0; p < n; ++p) u[p] = s[p] - (total - s[p]) / (n - 1);
}

// ---------------------------------------------------------------------------
// Situaciones (information sets), agrupando lo que es igual.
//
// Etiquetas de palo, desde el punto de vista del que decide:
//   0 = triunfo, 1 = palo de salida (si no es triunfo), 2 y 3 = los palos de
//   sus cartas que no son ni triunfo ni salida (2 = el de su carta mas alta),
//   4 = cualquier otro. Los palos que no son triunfo son intercambiables, asi
//   que solo importa su relacion con el triunfo, la salida y la mano propia.
// ---------------------------------------------------------------------------

enum Abstraction { kAbstractionB = 0, kExact = 1 };

struct Key {
  uint64_t a = 0, b = 0;
  bool operator==(const Key& o) const { return a == o.a && b == o.b; }
};
struct KeyHash {
  size_t operator()(const Key& k) const {
    return static_cast<size_t>(k.a * 0x9E3779B97F4A7C15ULL ^ (k.b + 0x632BE59BD9B4E019ULL) * 0xC2B2AE3D27D4EB4FULL);
  }
};

void SuitLabels(const Deal& d, int k, int lead_suit, int* lab) {
  for (int s = 0; s < 4; ++s) lab[s] = 4;
  lab[d.ts] = 0;
  if (lead_suit >= 0 && lead_suit != d.ts) lab[lead_suit] = 1;
  const int* h = d.hand[k];
  // Las cartas propias de palos todavia sin etiqueta: 2 para la mas alta.
  int hi = Rank(h[0]) >= Rank(h[1]) ? h[0] : h[1];
  int lo = hi == h[0] ? h[1] : h[0];
  if (lab[Suit(hi)] == 4) lab[Suit(hi)] = 2;
  if (lab[Suit(lo)] == 4) lab[Suit(lo)] = 3;
}

// Las dos cartas propias como (etiqueta, numero), ordenadas: 14 bits. Devuelve
// tambien el orden canonico (first = la que va primero).
uint64_t HandCode(const Deal& d, int k, const int* lab, int* first) {
  const int* h = d.hand[k];
  int c0 = lab[Suit(h[0])] * kRanks + Rank(h[0]);
  int c1 = lab[Suit(h[1])] * kRanks + Rank(h[1]);
  if (c0 <= c1) { *first = 0; } else { std::swap(c0, c1); *first = 1; }
  // Dos cartas de palos distintos con el mismo numero y etiquetas 2 y 3 son
  // intercambiables: quedan con el mismo codigo por como se etiqueta.
  return static_cast<uint64_t>(c0) << 7 | static_cast<uint64_t>(c1);
}

Key BidKey(const Deal& d, int k, const St& st) {
  int lab[4], first;
  SuitLabels(d, k, -1, lab);
  uint64_t prefix = 0;
  for (int j = 0; j < k; ++j) prefix = prefix * 3 + st.bids[j];
  Key key;
  key.a = (0ULL << 63) | static_cast<uint64_t>(k) << 50 | static_cast<uint64_t>(Rank(d.trump)) << 44 |
          HandCode(d, k, lab, &first) << 30 | prefix;
  return key;
}

Key PlayKey(const Deal& d, int k, const St& st, Abstraction abs, int* first) {
  int lead = k > 0 ? Suit(st.play1[0]) : -1, lab[4];
  SuitLabels(d, k, lead, lab);
  uint64_t bids = 0;
  for (int j = 0; j < d.n; ++j) bids = bids << 2 | st.bids[j];
  int winner = 0;
  for (int j = 1; j < k; ++j)
    if (Beats(st.play1[j], st.play1[winner], d.ts)) winner = j;
  Key key;
  key.a = (1ULL << 63) | static_cast<uint64_t>(k) << 58 | static_cast<uint64_t>(Rank(d.trump)) << 54 |
          HandCode(d, k, lab, first) << 40 | bids << 3 | static_cast<uint64_t>(winner);
  const int* h = d.hand[k];
  // Los palos "otros" se etiquetan 4 y 5 en el orden en que aparecen (asi se
  // distingue si dos cartas de esos palos son del mismo palo o no).
  int next_other = 4;
  for (int s = 0; s < 4; ++s) if (lab[s] == 4) lab[s] = -1;
  for (int j = 0; j < k; ++j) {
    int c = st.play1[j];
    if (lab[Suit(c)] < 0) lab[Suit(c)] = next_other++;
    int l = lab[Suit(c)], code;
    if (abs == kExact) {
      code = Rank(c);
    } else {
      // Agrupada: si tengo cartas de ese palo, cuantas de las mias son mas
      // bajas (1 + 0..2); si no tengo, el numero no se guarda (0).
      int mine = 0, below = 0;
      for (int i = 0; i < 2; ++i)
        if (Suit(h[i]) == Suit(c)) { ++mine; below += Rank(h[i]) < Rank(c); }
      code = mine ? 1 + below : 0;
    }
    key.b = key.b << 7 | static_cast<uint64_t>(l) << 4 | static_cast<uint64_t>(code);
  }
  return key;
}

struct Node {
  float regret[kMaxActions] = {0, 0, 0};
  float sum[kMaxActions] = {0, 0, 0};    // estrategia promedio (sin normalizar)
};

using Table = std::unordered_map<Key, Node, KeyHash>;

void RegretMatching(const Node& node, int m, double* sigma) {
  double tot = 0;
  for (int a = 0; a < m; ++a) tot += std::max(0.0f, node.regret[a]);
  for (int a = 0; a < m; ++a) sigma[a] = tot > 0 ? std::max(0.0f, node.regret[a]) / tot : 1.0 / m;
}

void AverageStrategy(const Node& node, int m, double* sigma) {
  double tot = 0;
  for (int a = 0; a < m; ++a) tot += node.sum[a];
  for (int a = 0; a < m; ++a) sigma[a] = tot > 0 ? node.sum[a] / tot : 1.0 / m;
}

int Sample(const double* sigma, int m, Rng& rng) {
  double x = rng.Uniform(), acc = 0;
  for (int a = 0; a < m - 1; ++a) if ((acc += sigma[a]) > x) return a;
  return m - 1;
}

// ---------------------------------------------------------------------------
// La heuristica de evaluation.heuristic_opponent, con 2 cartas.
// ---------------------------------------------------------------------------

const double kBidWeights[8][7] = {
    {}, {},
    {1.00, 0.95, 0.69, 0.58, 0.57, 0.50, 0.23},
    {1.00, 0.77, 0.56, 0.45, 0.37, 0.24, 0.06},
    {1.00, 0.64, 0.49, 0.31, 0.22, 0.16, 0.00},
    {0.99, 0.53, 0.38, 0.22, 0.15, 0.08, 0.00},
    {0.95, 0.47, 0.34, 0.14, 0.08, 0.03, 0.00},
    {0.92, 0.40, 0.27, 0.12, 0.05, 0.01, 0.00}};

double CardWeight(int c, int ts, int n) {
  const double* w = kBidWeights[std::min(std::max(n, 2), 7)];
  if (Suit(c) == ts) return Rank(c) >= 9 ? w[0] : w[1];      // J o mas alto (valor >= 11)
  int r = Rank(c);
  return r == 12 ? w[2] : r == 11 ? w[3] : r == 10 ? w[4] : r == 9 ? w[5] : w[6];
}

int HeuristicBid(const Deal& d, int k, const St& st) {
  double est = CardWeight(d.hand[k][0], d.ts, d.n) + CardWeight(d.hand[k][1], d.ts, d.n);
  int target = static_cast<int>(std::nearbyint(est));        // como round() de Python
  int legal[3], m = LegalBids(d.n, k, st, legal), best = legal[0];
  for (int i = 1; i < m; ++i)
    if (std::abs(legal[i] - target) < std::abs(best - target)) best = legal[i];
  return best;
}

int HeuristicPlay(const Deal& d, int k, const St& st) {
  int legal[2], m = LegalPlays(d, k, st, legal);
  // Bazas ganadas: en la primera baza todavia ninguna.
  bool wants = 0 < st.bids[k];
  auto low = [&](const int* cs, int cm) {     // min por (es triunfo, valor), el primero
    int best = cs[0];
    for (int i = 1; i < cm; ++i) {
      bool tb = Suit(best) == d.ts, tc = Suit(cs[i]) == d.ts;
      if (tc < tb || (tc == tb && Rank(cs[i]) < Rank(best))) best = cs[i];
    }
    return best;
  };
  if (k == 0) {
    int best = legal[0];
    for (int i = 1; i < m; ++i)
      if (wants ? Rank(legal[i]) > Rank(best) : Rank(legal[i]) < Rank(best)) best = legal[i];
    return best;
  }
  int cur = st.play1[0];
  for (int j = 1; j < k; ++j) if (Beats(st.play1[j], cur, d.ts)) cur = st.play1[j];
  int winners[2], wm = 0, losers[2], lm = 0;
  for (int i = 0; i < m; ++i) (Beats(legal[i], cur, d.ts) ? winners[wm++] : losers[lm++]) = legal[i];
  if (wants && wm) return low(winners, wm);
  return lm ? low(losers, lm) : low(legal, m);
}

// ---------------------------------------------------------------------------
// MCCFR con muestreo externo: para el jugador que se actualiza se recorren
// todas sus acciones; las de los demas y el reparto se muestrean. La
// estrategia promedio se acumula en los nodos de los demas.
// ---------------------------------------------------------------------------

struct Solver {
  int n;
  Abstraction abs;
  Table table;
  Rng rng;
  Solver(int n_, Abstraction a, uint64_t seed) : n(n_), abs(a), rng(seed) { table.reserve(1 << 20); }

  double Traverse(const Deal& d, St& st, int p) {
    if (st.nb < n) {
      int k = st.nb, legal[3], m = LegalBids(n, k, st, legal);
      Node& node = table[BidKey(d, k, st)];
      double sigma[3];
      RegretMatching(node, m, sigma);
      if (k == p) {
        double v[3], value = 0;
        for (int a = 0; a < m; ++a) {
          st.bids[k] = legal[a]; ++st.nb;
          v[a] = Traverse(d, st, p);
          --st.nb;
          value += sigma[a] * v[a];
        }
        for (int a = 0; a < m; ++a) node.regret[a] += static_cast<float>(v[a] - value);
        return value;
      }
      for (int a = 0; a < m; ++a) node.sum[a] += static_cast<float>(sigma[a]);
      st.bids[k] = legal[Sample(sigma, m, rng)]; ++st.nb;
      double v = Traverse(d, st, p);
      --st.nb;
      return v;
    }
    if (st.np < n) {
      int k = st.np, legal[2], m = LegalPlays(d, k, st, legal);
      if (m == 1) {
        st.play1[k] = legal[0]; ++st.np;
        double v = Traverse(d, st, p);
        --st.np;
        return v;
      }
      int first;
      Node& node = table[PlayKey(d, k, st, abs, &first)];
      int cards[2] = {d.hand[k][first], d.hand[k][1 - first]};   // orden canonico
      double sigma[2];
      RegretMatching(node, 2, sigma);
      if (k == p) {
        double v[2], value = 0;
        for (int a = 0; a < 2; ++a) {
          st.play1[k] = cards[a]; ++st.np;
          v[a] = Traverse(d, st, p);
          --st.np;
          value += sigma[a] * v[a];
        }
        for (int a = 0; a < 2; ++a) node.regret[a] += static_cast<float>(v[a] - value);
        return value;
      }
      for (int a = 0; a < 2; ++a) node.sum[a] += static_cast<float>(sigma[a]);
      st.play1[k] = cards[Sample(sigma, 2, rng)]; ++st.np;
      double v = Traverse(d, st, p);
      --st.np;
      return v;
    }
    double u[kMaxPlayers];
    Utilities(d, st, u);
    return u[p];
  }

  void Iterate() {
    for (int p = 0; p < n; ++p) {
      Deal d;
      DealCards(d, n, rng);
      St st;
      Traverse(d, st, p);
    }
  }
};

// ---------------------------------------------------------------------------
// Jugar con estrategias fijas (para evaluar).
// ---------------------------------------------------------------------------

enum PolicyKind { kAverage, kHeuristic, kUniform };

struct Player {
  PolicyKind kind;
  const Table* table;
  Abstraction abs;
};

struct Misses { long long lookups = 0, misses = 0; };

int ChooseBid(const Player& pl, const Deal& d, int k, const St& st, Rng& rng, Misses& ms) {
  int legal[3], m = LegalBids(d.n, k, st, legal);
  if (pl.kind == kHeuristic) return HeuristicBid(d, k, st);
  double sigma[3];
  for (int a = 0; a < m; ++a) sigma[a] = 1.0 / m;
  if (pl.kind == kAverage) {
    ++ms.lookups;
    auto it = pl.table->find(BidKey(d, k, st));
    if (it == pl.table->end()) ++ms.misses; else AverageStrategy(it->second, m, sigma);
  }
  return legal[Sample(sigma, m, rng)];
}

int ChoosePlay(const Player& pl, const Deal& d, int k, const St& st, Rng& rng, Misses& ms) {
  int legal[2], m = LegalPlays(d, k, st, legal);
  if (m == 1) return legal[0];
  if (pl.kind == kHeuristic) return HeuristicPlay(d, k, st);
  int first;
  Key key = PlayKey(d, k, st, pl.abs, &first);
  int cards[2] = {d.hand[k][first], d.hand[k][1 - first]};
  double sigma[2] = {0.5, 0.5};
  if (pl.kind == kAverage) {
    ++ms.lookups;
    auto it = pl.table->find(key);
    if (it == pl.table->end()) ++ms.misses; else AverageStrategy(it->second, 2, sigma);
  }
  return cards[Sample(sigma, 2, rng)];
}

void PlayRound(const std::vector<Player>& players, const Deal& d, St& st, Rng& rng, Misses& ms) {
  for (int k = 0; k < d.n; ++k) { st.bids[k] = ChooseBid(players[k], d, k, st, rng, ms); ++st.nb; }
  for (int k = 0; k < d.n; ++k) { st.play1[k] = ChoosePlay(players[k], d, k, st, rng, ms); ++st.np; }
}

// ---------------------------------------------------------------------------
// Mejor respuesta: cuanto ganaria la posicion p cambiando su forma de jugar
// mientras los demas siguen su estrategia promedio (dentro de las mismas
// situaciones agrupadas). Con los mismos repartos y los mismos sorteos para
// los demas (numeros aleatorios comunes) en todas las alternativas de p:
//   1. para cada situacion de jugada de p, el valor de cada carta;
//   2. para cada situacion de canto de p, el valor de cada canto jugando
//      despues lo mejor del paso 1.
// La ganancia es el valor de la mejor respuesta menos el de la estrategia.
// ---------------------------------------------------------------------------

// Uniforme determinista para el reparto `deal`, la decision de la posicion k
// en la fase `phase`: los demas sortean igual en todas las alternativas de p.
double Common(uint64_t seed, uint64_t deal, int k, int phase) {
  Rng r(seed * 0x9E3779B97F4A7C15ULL ^ deal * 0xD1B54A32D192ED03ULL ^ (k * 2 + phase + 1) * 0xABCDEF12345ULL);
  return r.Uniform();
}

int SampleU(const double* sigma, int m, double x) {
  double acc = 0;
  for (int a = 0; a < m - 1; ++a) if ((acc += sigma[a]) > x) return a;
  return m - 1;
}

using QTable = std::unordered_map<Key, std::array<double, kMaxActions>, KeyHash>;

struct Deviation {
  int bid = -1;          // canto de p (-1: el de su estrategia)
  int play = -1;         // carta de p, en orden canonico (-1: la de su estrategia o la mejor)
  const QTable* best_play = nullptr;   // si no es null, p juega lo mejor segun esta tabla
};

// Juega la ronda con los demas siguiendo la tabla. Devuelve el reward de p y,
// si p tuvo que elegir carta, la situacion (play_key) y has_play = true.
double Simulate(const Deal& d, int p, const Table& table, Abstraction abs, uint64_t seed,
                uint64_t deal_id, const Deviation& dev, Key* play_key, bool* has_play) {
  St st;
  *has_play = false;
  for (int k = 0; k < d.n; ++k) {
    int legal[3], m = LegalBids(d.n, k, st, legal);
    if (k == p && dev.bid >= 0) {
      st.bids[k] = dev.bid;
    } else {
      double sigma[3];
      for (int a = 0; a < m; ++a) sigma[a] = 1.0 / m;
      auto it = table.find(BidKey(d, k, st));
      if (it != table.end()) AverageStrategy(it->second, m, sigma);
      st.bids[k] = legal[SampleU(sigma, m, Common(seed, deal_id, k, 0))];
    }
    ++st.nb;
  }
  for (int k = 0; k < d.n; ++k) {
    int legal[2], m = LegalPlays(d, k, st, legal);
    if (m == 1) { st.play1[k] = legal[0]; ++st.np; continue; }
    int first;
    Key key = PlayKey(d, k, st, abs, &first);
    int cards[2] = {d.hand[k][first], d.hand[k][1 - first]};
    double sigma[2] = {0.5, 0.5};
    auto it = table.find(key);
    if (it != table.end()) AverageStrategy(it->second, 2, sigma);
    int a = SampleU(sigma, 2, Common(seed, deal_id, k, 1));
    if (k == p) {
      *has_play = true;
      *play_key = key;
      if (dev.play >= 0) {
        a = dev.play;
      } else if (dev.best_play) {
        auto q = dev.best_play->find(key);
        if (q != dev.best_play->end()) a = q->second[1] > q->second[0];
      }
    }
    st.play1[k] = cards[a];
    ++st.np;
  }
  double u[kMaxPlayers];
  Utilities(d, st, u);
  return u[p];
}

// Los cantos de las posiciones antes de p (no dependen de lo que haga p), con
// los mismos sorteos que Simulate.
void BidsBefore(const Deal& d, int p, const Table& table, uint64_t seed, uint64_t deal_id, St& st) {
  for (int k = 0; k < p; ++k) {
    int legal[3], m = LegalBids(d.n, k, st, legal);
    double sigma[3];
    for (int i = 0; i < m; ++i) sigma[i] = 1.0 / m;
    auto it = table.find(BidKey(d, k, st));
    if (it != table.end()) AverageStrategy(it->second, m, sigma);
    st.bids[k] = legal[SampleU(sigma, m, Common(seed, deal_id, k, 0))];
    ++st.nb;
  }
}

void BestResponse(const Args& a, const Table& table, Abstraction abs) {
  std::vector<double> gains, holdouts;
  for (int p = 0; p < a.n; ++p) {
    QTable qplay, qbid;
    Key key;
    bool has;
    // 1. Valor de cada carta en cada situacion de jugada de p (para cada canto suyo).
    for (long long g = 0; g < a.games; ++g) {
      Rng rng(a.seed * 1000003ULL + g);
      Deal d;
      DealCards(d, a.n, rng);
      St before;
      BidsBefore(d, p, table, a.seed, g, before);
      int legal[3], m = LegalBids(a.n, p, before, legal);
      for (int i = 0; i < m; ++i) {
        Deviation dev;
        dev.bid = legal[i];
        dev.play = 0;
        double v0 = Simulate(d, p, table, abs, a.seed, g, dev, &key, &has);
        if (!has) continue;
        dev.play = 1;
        double v1 = Simulate(d, p, table, abs, a.seed, g, dev, &key, &has);
        auto& q = qplay[key];
        q[0] += v0;
        q[1] += v1;
      }
    }
    // 2. Valor de cada canto jugando despues lo mejor; y el de la estrategia.
    double base = 0;
    for (long long g = 0; g < a.games; ++g) {
      Rng rng(a.seed * 1000003ULL + g);
      Deal d;
      DealCards(d, a.n, rng);
      Deviation dev;
      base += Simulate(d, p, table, abs, a.seed, g, dev, &key, &has);
      St before;
      BidsBefore(d, p, table, a.seed, g, before);
      int legal[3], m = LegalBids(a.n, p, before, legal);
      auto& q = qbid[BidKey(d, p, before)];
      dev.best_play = &qplay;
      for (int i = 0; i < m; ++i) {
        dev.bid = legal[i];
        q[i] += Simulate(d, p, table, abs, a.seed, g, dev, &key, &has);
      }
      for (int i = m; i < kMaxActions; ++i) q[i] = -1e300;
    }
    double best = 0;
    for (const auto& kv : qbid) best += *std::max_element(kv.second.begin(), kv.second.end());
    double gain = (best - base) / a.games;
    // 3. La misma desviacion (mejor canto y mejor jugada elegidos arriba),
    // medida en otros repartos: sin el ruido a favor de elegir lo mejor sobre
    // una muestra, es una cota por abajo de la ganancia.
    double dev_sum = 0, base_sum = 0;
    for (long long g = 0; g < a.games; ++g) {
      uint64_t id = a.games + g;           // repartos que no se usaron para elegir
      Rng rng(a.seed * 1000003ULL + id);
      Deal d;
      DealCards(d, a.n, rng);
      Deviation dev;
      base_sum += Simulate(d, p, table, abs, a.seed, id, dev, &key, &has);
      St before;
      BidsBefore(d, p, table, a.seed, id, before);
      int legal[3], m = LegalBids(a.n, p, before, legal);
      auto q = qbid.find(BidKey(d, p, before));
      if (q != qbid.end()) dev.bid = legal[std::max_element(q->second.begin(), q->second.begin() + m) - q->second.begin()];
      dev.best_play = &qplay;
      dev_sum += Simulate(d, p, table, abs, a.seed, id, dev, &key, &has);
    }
    double holdout = (dev_sum - base_sum) / a.games;
    gains.push_back(gain);
    holdouts.push_back(holdout);
  }
  // {"gain_in_sample": [...], "mean_in_sample": x, "gain_holdout": [...], "mean_holdout": y}
  auto field = [](const char* name, const std::vector<double>& v) {
    std::string out = std::string("\"gain_") + name + "\": [";
    double mean = 0;
    char buf[64];
    for (size_t i = 0; i < v.size(); ++i) {
      std::snprintf(buf, sizeof(buf), "%s%.5f", i ? ", " : "", v[i]);
      out += buf;
      mean += v[i] / v.size();
    }
    std::snprintf(buf, sizeof(buf), "], \"mean_%s\": %.5f", name, mean);
    return out + buf;
  };
  std::printf("{%s, %s}\n", field("in_sample", gains).c_str(), field("holdout", holdouts).c_str());
}

// Exporta la estrategia promedio de cada situacion, una por linea:
//   bid  k numero_triunfo carta1 carta2 cantos_previos(base 3)         p0 p1 p2 peso
//   play k numero_triunfo carta1 carta2 cantos(2 bits c/u) gana jugadas p0 p1 - peso
// donde carta = etiqueta * 13 + numero (etiquetas en SuitLabels), jugadas es el
// codigo de las cartas de la baza (7 bits cada una: etiqueta << 4 | codigo) y
// p0, p1 son las probabilidades de jugar carta1 y carta2. peso = cuanto se
// visito (suma de la estrategia acumulada).
void Dump(const Table& table, const std::string& path) {
  FILE* f = std::fopen(path.c_str(), "w");
  for (const auto& kv : table) {
    const Key& k = kv.first;
    const Node& node = kv.second;
    double w = node.sum[0] + node.sum[1] + node.sum[2];
    if (w <= 0) continue;
    if (!(k.a >> 63)) {
      std::fprintf(f, "bid\t%llu\t%llu\t%llu\t%llu\t%llu\t%.4f\t%.4f\t%.4f\t%.1f\n",
                   (unsigned long long)(k.a >> 50 & 7), (unsigned long long)(k.a >> 44 & 15),
                   (unsigned long long)(k.a >> 37 & 127), (unsigned long long)(k.a >> 30 & 127),
                   (unsigned long long)(k.a & ((1ULL << 30) - 1)),
                   node.sum[0] / w, node.sum[1] / w, node.sum[2] / w, w);
    } else {
      std::fprintf(f, "play\t%llu\t%llu\t%llu\t%llu\t%llu\t%llu\t%llu\t%.4f\t%.4f\t-\t%.1f\n",
                   (unsigned long long)(k.a >> 58 & 7), (unsigned long long)(k.a >> 54 & 15),
                   (unsigned long long)(k.a >> 47 & 127), (unsigned long long)(k.a >> 40 & 127),
                   (unsigned long long)(k.a >> 3 & ((1ULL << 37) - 1)), (unsigned long long)(k.a & 7),
                   (unsigned long long)k.b, node.sum[0] / w, node.sum[1] / w, w);
    }
  }
  std::fclose(f);
}

// ---------------------------------------------------------------------------
// Guardar y cargar la tabla (para seguir en otra sesion).
// ---------------------------------------------------------------------------

void Save(const Solver& s, const std::string& path, long long iters) {
  std::ofstream f(path, std::ios::binary);
  int32_t header[3] = {s.n, static_cast<int32_t>(s.abs), 1};
  f.write(reinterpret_cast<const char*>(header), sizeof(header));
  f.write(reinterpret_cast<const char*>(&iters), sizeof(iters));
  uint64_t size = s.table.size();
  f.write(reinterpret_cast<const char*>(&size), sizeof(size));
  for (const auto& [k, v] : s.table) {
    f.write(reinterpret_cast<const char*>(&k), sizeof(k));
    f.write(reinterpret_cast<const char*>(&v), sizeof(v));
  }
}

long long Load(Solver& s, const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) { std::fprintf(stderr, "No se pudo abrir %s\n", path.c_str()); std::exit(1); }
  int32_t header[3];
  long long iters;
  uint64_t size;
  f.read(reinterpret_cast<char*>(header), sizeof(header));
  f.read(reinterpret_cast<char*>(&iters), sizeof(iters));
  f.read(reinterpret_cast<char*>(&size), sizeof(size));
  if (header[0] != s.n || header[1] != s.abs) {
    std::fprintf(stderr, "La tabla es de n=%d abstraccion=%d\n", header[0], header[1]);
    std::exit(1);
  }
  s.table.reserve(size);
  for (uint64_t i = 0; i < size; ++i) {
    Key k;
    Node v;
    f.read(reinterpret_cast<char*>(&k), sizeof(k));
    f.read(reinterpret_cast<char*>(&v), sizeof(v));
    s.table.emplace(k, v);
  }
  return iters;
}

// ---------------------------------------------------------------------------


Args Parse(int argc, char** argv) {
  Args a;
  if (argc < 2) { std::fprintf(stderr, "uso: round2 solve|eval|trace [opciones]\n"); std::exit(1); }
  a.cmd = argv[1];
  for (int i = 2; i + 1 < argc; i += 2) {
    std::string k = argv[i], v = argv[i + 1];
    if (k == "--n") a.n = std::stoi(v);
    else if (k == "--iters") a.iters = std::stoll(v);
    else if (k == "--games") a.games = std::stoll(v);
    else if (k == "--seed") a.seed = std::stoull(v);
    else if (k == "--abstraction") a.abstraction = v;
    else if (k == "--load") a.load = v;
    else if (k == "--save") a.save = v;
    else if (k == "--policies") a.policies = v;
    else if (k == "--report_every") a.report_every = std::stoll(v);
    else { std::fprintf(stderr, "opcion desconocida %s\n", k.c_str()); std::exit(1); }
  }
  if (a.n < 2 || a.n > kMaxPlayers) { std::fprintf(stderr, "--n entre 2 y 7\n"); std::exit(1); }
  return a;
}

Abstraction ParseAbstraction(const std::string& s) {
  if (s == "b") return kAbstractionB;
  if (s == "exact") return kExact;
  std::fprintf(stderr, "--abstraction b|exact\n");
  std::exit(1);
}

// Evaluar: jugar `games` rondas con una estrategia por posicion; imprime el
// reward medio de cada posicion y su error estandar.
void Evaluate(const Args& a, const Table& table, Abstraction abs) {
  // Una politica por posicion (avg, heur o uniform); si hay menos, se repite la ultima.
  std::vector<Player> players;
  std::string pols = a.policies;
  for (size_t pos = 0; pos <= pols.size();) {
    size_t next = pols.find(',', pos);
    if (next == std::string::npos) next = pols.size();
    std::string p = pols.substr(pos, next - pos);
    PolicyKind kind;
    if (p == "avg") kind = kAverage;
    else if (p == "heur") kind = kHeuristic;
    else if (p == "uniform") kind = kUniform;
    else { std::fprintf(stderr, "politica desconocida: %s\n", p.c_str()); std::exit(1); }
    players.push_back({kind, &table, abs});
    pos = next + 1;
  }
  while (players.size() < static_cast<size_t>(a.n)) players.push_back(players.back());
  players.resize(a.n);
  Rng rng(a.seed);
  Misses ms;
  std::vector<double> sum(a.n, 0), sq(a.n, 0);
  for (long long g = 0; g < a.games; ++g) {
    Deal d;
    DealCards(d, a.n, rng);
    St st;
    PlayRound(players, d, st, rng, ms);
    double u[kMaxPlayers];
    Utilities(d, st, u);
    for (int p = 0; p < a.n; ++p) { sum[p] += u[p]; sq[p] += u[p] * u[p]; }
  }
  std::printf("{\"value\": [");
  for (int p = 0; p < a.n; ++p) std::printf("%s%.5f", p ? ", " : "", sum[p] / a.games);
  std::printf("], \"se\": [");
  for (int p = 0; p < a.n; ++p) {
    double mean = sum[p] / a.games, var = sq[p] / a.games - mean * mean;
    std::printf("%s%.5f", p ? ", " : "", std::sqrt(std::max(var, 0.0) / a.games));
  }
  std::printf("], \"miss_rate\": %.6f}\n", ms.lookups ? static_cast<double>(ms.misses) / ms.lookups : 0.0);
}

// Traza de rondas con jugadas al azar, mas lo que haria la heuristica en cada
// decision, para compararlo con el motor de Python (tests/test_round2.py).
void Trace(const Args& a) {
  Rng rng(a.seed);
  for (long long g = 0; g < a.games; ++g) {
    Deal d;
    DealCards(d, a.n, rng);
    St st;
    std::printf("{\"trump\": %d, \"hands\": [", d.trump);
    for (int p = 0; p < a.n; ++p) std::printf("%s[%d, %d]", p ? ", " : "", d.hand[p][0], d.hand[p][1]);
    std::printf("], \"bids\": [");
    for (int k = 0; k < a.n; ++k) {
      int legal[3], m = LegalBids(a.n, k, st, legal), h = HeuristicBid(d, k, st);
      int b = legal[rng.Below(m)];
      std::printf("%s{\"legal\": [", k ? ", " : "");
      for (int i = 0; i < m; ++i) std::printf("%s%d", i ? ", " : "", legal[i]);
      std::printf("], \"chosen\": %d, \"heuristic\": %d}", b, h);
      st.bids[k] = b; ++st.nb;
    }
    std::printf("], \"plays\": [");
    for (int k = 0; k < a.n; ++k) {
      int legal[2], m = LegalPlays(d, k, st, legal), h = HeuristicPlay(d, k, st);
      int c = legal[rng.Below(m)];
      std::printf("%s{\"legal\": [", k ? ", " : "");
      for (int i = 0; i < m; ++i) std::printf("%s%d", i ? ", " : "", legal[i]);
      std::printf("], \"chosen\": %d, \"heuristic\": %d}", c, h);
      st.play1[k] = c; ++st.np;
    }
    int tricks[kMaxPlayers];
    Tricks(d, st, tricks);
    std::printf("], \"scores\": [");
    for (int p = 0; p < a.n; ++p) std::printf("%s%d", p ? ", " : "", Score(st.bids[p], tricks[p]));
    std::printf("]}\n");
  }
}

}  // namespace

int main(int argc, char** argv) {
  Args a = Parse(argc, argv);
  if (a.cmd == "trace") { Trace(a); return 0; }
  Abstraction abs = ParseAbstraction(a.abstraction);
  Solver s(a.n, abs, a.seed);
  long long done = a.load.empty() ? 0 : Load(s, a.load);
  if (a.cmd == "eval") { Evaluate(a, s.table, abs); return 0; }
  if (a.cmd == "br") { BestResponse(a, s.table, abs); return 0; }
  if (a.cmd == "dump") { Dump(s.table, a.save); return 0; }
  if (a.cmd != "solve") { std::fprintf(stderr, "comando %s\n", a.cmd.c_str()); return 1; }
  auto t0 = std::chrono::steady_clock::now();
  for (long long it = 1; it <= a.iters; ++it) {
    s.Iterate();
    if (a.report_every && it % a.report_every == 0) {
      double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
      std::fprintf(stderr, "iteracion %lld (%lld en total): %zu situaciones, %.0f s\n", it,
                   done + it, s.table.size(), secs);
      if (!a.save.empty()) Save(s, a.save, done + it);
    }
  }
  if (!a.save.empty()) Save(s, a.save, done + a.iters);
  std::fprintf(stderr, "listo: %lld iteraciones en total, %zu situaciones\n", done + a.iters, s.table.size());
  return 0;
}
