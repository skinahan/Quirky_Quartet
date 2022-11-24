def to_prompt(description, prompt_type='Default'):
    ''' Create default, chain of thought, or least-to-most prompts '''
    match prompt_type:
        case 'Default':
            return f'"""Write a python function to {description}"""\ndef'
        case extended:
            n_components = description.count(' then ') + 1
            components = description.split(' then ')
            component_list = ''
            for c, component in enumerate(components):
                component_list += f'{c+1}. {component}\n'
            if 'Least' in extended:
                fstr = f'"""Write a python function with {n_components} components. These components should be composed step by step:\n{component_list}"""\ndef'
            else:
                fstr = f'"""Write a python function to:\n{component_list}'
            return fstr

