from .formula_integration import FormulaIntegration

# 单例模式的公式集成器实例
_formula_integration_instance = None

def get_formula_integration():
    """
    获取FormulaIntegration的单例实例
    :return: FormulaIntegration实例
    """
    global _formula_integration_instance
    if _formula_integration_instance is None:
        _formula_integration_instance = FormulaIntegration()
    return _formula_integration_instance 