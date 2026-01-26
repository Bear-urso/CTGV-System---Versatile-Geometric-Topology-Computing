# CTGV System V1.5
# Versatile Geometric Topology Computing

[![License: LIACC](https://github.com/Bear-urso/LICENSE-LIACC-)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyPI version](https://badge.fury.io/py/ctgv-system.svg)](https://pypi.org/project/ctgv-system/)

**Cognitive Engine for Distributed Topological Processing**

The CTGV System is a computational architecture, developed using an algorithm adapted from CTCE - Electromagnetic Field Topology Computation:
Modeled in Python, it is treated here as "Proto-Gebit", formulated through graphs and pictograms. It serves as a semantic flow regulator also applicable to AI with explicit decision-making, with traceability and architectural neutrality. The system provides a basis for cognitive governance and didactic emulation through geometric topology.

And with a filter of: **how to think**, and not just **what to answer** in a self-directed manner similar to the "life game of John Conway", but with an approach harmonizing graphs, geometry, fractals and versatile topologies to make decisions in a traceable and auditable way.

## ✨ Key Features (potential)

- 🚀 **Distributed Processing**: Scale to millions of nodes with intelligent graph partitioning
- 🧠 **Topological Cognition**: Geometric approach to cognitive processing
- ⚡ **GPU Acceleration**: CUDA/CuPy support for high-performance computing
- 🔄 **Real-time Streaming**: Continuous data processing pipelines
- 🎯 **Auto-scaling**: Dynamic resource management based on workload
- 📊 **Advanced Monitoring**: Comprehensive system metrics and alerting

## Installation

### Basic Installation
```bash
pip install ctgv-system
```

### With GPU Support
```bash
pip install ctgv-system[gpu]
```

### With Machine Learning Features
```bash
pip install ctgv-system[ml]
```

### Development Installation
```bash
git clone https://github.com/Bear-urso/CTGV-System---Versatile-Geometric-Topology-Computing.git
cd CTGV-System---Versatile-Geometric-Topology-Computing
pip install -e .[dev]
```

## Quick Start

### Basic Usage
```python
from ctgv import Shape, Gebit, CTGVEngine

# Create gebits
origin = Gebit(Shape.ORIGIN, intensity=1.0, label="Source")
flow = Gebit(Shape.FLOW, intensity=0.0, label="Channel")

# Connect them
origin.connect_to(flow, 0.9)

# Propagate signal
engine = CTGVEngine()
result = engine.propagate([origin])

print(f"Converged: {result['converged']}")
print(f"Final states: {result['final_states']}")
```

### Distributed Processing
```python
from ctgv import DistributedCTGVEngine

# Create large network
nodes = [Gebit(Shape.FLOW, intensity=0.1) for _ in range(10000)]

# Distributed processing
distributed_engine = DistributedCTGVEngine(num_workers=8, use_gpu=True)
result = distributed_engine.distributed_propagate(nodes)

print(f"Processed {result['total_nodes_processed']} nodes")
```

### Real-time Streaming
```python
from ctgv import StreamProcessingPipeline

# Create streaming pipeline
pipeline = StreamProcessingPipeline(batch_size=100)
pipeline.start(num_workers=4)

# Process streaming data
for batch in streaming_data_batches:
    pipeline.feed(batch)
    results = pipeline.get_results(timeout=1.0)
    # Process results...

pipeline.stop()
```

## Architecture

### Core Components

- **🧬 Gebites**: Fundamental units with geometric shapes and vector fields
- **📐 Shapes**: 10 pictographic forms (Origin, Flow, Decisor, Memory, Resonator, etc.)
- **🌊 Topological Propagation**: Signal flow through weighted connections
- **⚛️ Field Interference**: Vector-based electromagnetic interactions
- **📏 Geometric Constraints**: Shape-specific connection rules

### Distributed Architecture

- **🕸️ Graph Partitioning**: Spectral, community detection, and geometric algorithms
- **⚖️ Load Balancing**: Intelligent distribution across processing units
- **🔄 Synchronization**: Cross-partition state coordination
- **🚀 GPU Acceleration**: CUDA kernels for compute-intensive operations
- **📊 Auto-scaling**: Dynamic worker management based on system metrics

## API Reference

### Core Classes

- `ctgv.shapes`: Shape definitions and geometric constraints
- `ctgv.vector_field`: Electromagnetic field representation
- `ctgv.gebit`: Fundamental CTGV unit
- `ctgv.engine`: Core propagation engine
- `ctgv.modeler`: Data encoding/decoding
- `ctgv.arbiter`: Temporal Binding Arbiter for ambiguity resolution
- `ctgv.clarification`: Pattern clarification engine

### Distributed Classes

- `ctgv.DistributedCTGVEngine`: Distributed processing engine
- `ctgv.StreamProcessingPipeline`: Real-time streaming pipeline
- `ctgv.GPUAccelerator`: GPU acceleration utilities
- `ctgv.HyperscaleCTGVSYSTEM`: Complete hyperscale system
- `ctgv.AutoScalingManager`: Dynamic scaling management
- `ctgv.SystemMonitor`: System monitoring and metrics

## Examples

See the `examples/` directory for comprehensive usage examples:

- `example.py`: Basic CTGV system demonstration
- `distributed_demo.py`: Distributed architecture showcase
- `performance_benchmark.py`: Performance analysis tools

## Performance Benchmarks

| Configuration | Network Size | Processing Time | Efficiency |
|---------------|-------------|-----------------|------------|
| Basic Engine | 2,000 nodes | ~0.12s | 16,667 nodes/s |
| Distributed | 2,000 nodes | ~0.08s | 25,000 nodes/s |
| GPU Accelerated | 10,000 nodes | ~0.15s | 66,667 nodes/s |
| Hyperscale | 50,000+ nodes | ~0.5s | 100,000+ nodes/s |

## Target Applications

### 🧠 Research & Academia
- Cognitive modeling and neuroscience
- AI safety and alignment research
- Complex systems simulation
- Mathematical topology applications

### 🏢 Enterprise & Industry
- Large-scale graph analytics
- Real-time recommendation systems
- Fraud detection networks
- Supply chain optimization

### 🎮 Gaming & Simulation
- Procedural content generation
- AI-driven NPC behaviors
- Complex ecosystem simulation
- Real-time strategy systems

### 🔬 Scientific Computing
- Molecular interaction modeling
- Epidemiological simulation
- Financial network analysis
- Climate system modeling

## Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### Development Setup
```bash
git clone https://github.com/Bear-urso/CTGV-System---Versatile-Geometric-Topology-Computing.git
cd CTGV-System---Versatile-Geometric-Topology-Computing
pip install -e .[dev]
pytest tests/
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use CTGV in your research, please cite:

```bibtex
@software{ctgv_system,
  title = {CTGV System - Versatile Geometric Topology Computing},
  author = {DOS SANTOS PORTO, BEGNOMAR},
  url = {https://github.com/Bear-urso/CTGV-System---Versatile-Geometric-Topology-Computing},
  year = {2024}
}
```

## Roadmap

- [ ] Enhanced GPU kernel optimizations
- [ ] Kubernetes deployment support
- [ ] Web-based visualization interface
- [ ] Integration with major ML frameworks
- [ ] Advanced graph partitioning algorithms
- [ ] Real-time collaborative processing

## Support

- 📖 [Documentation](https://github.com/Bear-urso/CTGV-System---Versatile-Geometric-Topology-Computing)
- 🐛 [Issue Tracker](https://github.com/Bear-urso/CTGV-System---Versatile-Geometric-Topology-Computing/issues)
- 💬 [Discussions](https://github.com/Bear-urso/CTGV-System---Versatile-Geometric-Topology-Computing/discussions)

---

**CTGV System**: *Revolutionizing cognitive computing through geometric topology* 🚀
- `ctgv.clarification_mechanism`: Advanced clarification for tie-breaking and training
- `ctgv.utils`: Utility functions and visualization

## Examples

See `example.py` for complete demonstrations including:
- Simple network propagation
- Ambiguity resolution with Temporal Binding Arbiter
- 2D pattern processing
- Feedback loops and resonance
- **Advanced Clarification Mechanism**: Tie-breaking and entropy reduction

## Como Usar

### Launcher (Recomendado)
```bash
python launcher.py --gui      # Interface gráfica interativa
python launcher.py --example  # Exemplo em linha de comando
```

### Uso Direto
```bash
python gui.py                 # Interface gráfica
python example.py             # Demonstração completa
```

## Clarification Engine

O sistema inclui um **ClarificationEngine** avançado para esclarecer decisões e reduzir ambiguidades:

### Clarificação de Decisões
```python
from ctgv import ClarificationEngine, CTGVEngine

clarifier = ClarificationEngine(CTGVEngine())
result = clarifier.clarify_decision(decision_options, context_factors)
```

### Clarificação de Padrões
```python
# Focos disponíveis: 'symmetry', 'structure', 'noise'
clarified = clarifier.clarify_pattern(pattern_2d, 'symmetry')
```

### Funcionalidades
- **Redução de Ambiguidade**: Processo iterativo para esclarecer decisões
- **Análise de Contexto**: Incorpora fatores contextuais na clarificação
- **Clarificação de Padrões**: Melhora simetria, estrutura e reduz ruído
- **Métricas de Qualidade**: Avalia confiança e melhoria de coerência

## Advanced Clarification Mechanism

O sistema agora inclui um **ClarificationMechanism** avançado para desempate e treinamento através de intervenção pontual:

### Detecção de Estados Ambíguos
```python
from ctgv import ClarificationMechanism, CTGVEngine

engine = CTGVEngine()
clarifier = ClarificationMechanism(engine)

# Detecta ambiguidades em lista de gebits
ambiguous_check = clarifier.check_ambiguous_state(gebit_list)
if ambiguous_check:
    print(f"Entropia detectada: {ambiguous_check['entropy_level']:.3f}")
```

### Arbiter Aprimorado com Clarificação
```python
from ctgv import EnhancedTemporalBindingArbiter

# Arbiter com mecanismo de clarificação integrado
enhanced_arbiter = EnhancedTemporalBindingArbiter(engine, binding_threshold=0.7)

# Resolve ambiguidades com intervenção automática
result = enhanced_arbiter.resolve_ambiguity(competing_gebits)
```

### Funcionalidades Avançadas
- **Cálculo de Entropia**: Shannon entropy para medir ambiguidade
- **Viés Multi-Critérios**: Avaliação baseada em estabilidade histórica, coerência de campo, força de conexão e centralidade geométrica
- **Intervenção Pontual**: Aplicação direcionada para reduzir entropia
- **Integração com TBA**: Temporal Binding Arbiter aprimorado com clarificação
- **Histórico de Aprendizado**: Rastreamento de intervenções aplicadas

### Demonstração Completa
```bash
python example.py  # Inclui demonstração do mecanismo avançado
```

Para facilitar o uso, incluímos uma interface gráfica completa:

```bash
python gui.py
```

### Funcionalidades da GUI:
- **Criação Visual de Gebits**: Selecione formas e configure parâmetros
- **Conexões Interativas**: Conecte Gebits com pesos personalizáveis
- **Visualização da Rede**: Veja a topologia em tempo real
- **Controle de Simulação**: Ajuste parâmetros do engine
- **Processamento de Padrões**: Insira matrizes 2D para processamento
- **Resultados em Tempo Real**: Visualize estados finais e métricas

### Captura de Tela da Interface:
- **Painel Esquerdo**: Lista de Gebits criados
- **Centro**: Visualização gráfica da rede
- **Abas Direitas**:
  - Rede: Topologia visual
  - Parâmetros: Configurações do engine
  - Resultados: Saída da simulação

## Verificação do Ambiente

Antes de usar, verifique se tudo está instalado:

```bash
python check_env.py
```

## Arquivos do Projeto

- `ctgv/`: Pacote principal do sistema
- `gui.py`: Interface gráfica completa
- `example.py`: Demonstrações em linha de comando
- `launcher.py`: Inicializador com opções
- `check_env.py`: Verificador de dependências
- `tests/`: Testes automatizados
- `requirements.txt`: Dependências Python

## NOTE: I AM BREAKING RIGID PATTERNS OF FORMALITY AND BUREAUCRACY.
IN RETURN, I OFFER TRANSPARENCY AND HONESTY.
THIS IS HOW OPEN AND FREE SCIENCE IS BUILT.
WITH PROOF OF COMPETENCE.
https://zenodo.org/records/18360864
https://doi.org/10.5281/zenodo.18360864

## License

Commercial use requires explicit license from the Creator
Creator: BEGNOMAR DOS SANTOS PORTO (@begnomar)
ORCID: 0009-0002-6109-7443

Declaration of Authorship, Intellectual Property and Rights Under Creation.
SHA256 Hash: 38669f499f7eb9d9bcb43838c4948db7f15b9b53d0ba6a243f8faed52715eb20
Linked to Bitcoin Block 933302, mined on January 21, 2026 at 11:59:44 UTC
