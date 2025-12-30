---
layout: learning-post-layout
title: "Python Language Rules"
date: 2025-12-30
lang: en
topic_url: /learning/python.html
translate_url: /cn/learning/python_guidelines/python_language_conventions.html
mathjax: false
---

## Lint

> 💡 **Tip**: Run pylint over your code using [pylintrc](https://google.github.io/styleguide/pylintrc).

**Definition**:
pylint is a tool for finding bugs and style problems in Python source code. It finds problems that are typically caught by a compiler for less dynamic languages like C and C++. Due to the dynamic nature of Python, some warnings may be incorrect; however, spurious warnings should be fairly infrequent.

**Pros**:
Catches easy-to-miss errors like typos, using-vars-before-assignment, etc.

**Cons**:
pylint isn't perfect. To take advantage of it, we'll sometimes need to write around it, suppress its warnings, or improve it.

**Decision**:
Be sure to run pylint on your code.

Suppress warnings if they are inappropriate so that other issues are not hidden. You can suppress warnings by using a line-level comment:

```python
def do_PUT(self):  # WSGI interface name, so pylint: disable=invalid-name
    ...
```

pylint warnings are each identified by a symbolic name (e.g. `empty-docstring`). Google-specific warnings start with `g-`.

If the reason for the suppression is not clear from the symbolic name, add an explanation.

Suppressing in this way has the advantage that we can easily search for the suppressions and re-evaluate them.

You can get a list of pylint warnings by using `pylint --list-msgs`. To get more details on a particular message, use `pylint --help-msg=invalid-name`.

Prefer `pylint: disable` to the older form `pylint: disable-msg`.

Unused argument warnings can be suppressed by deleting the variables at the beginning of the function. Always include a comment explaining why they are unused. "Unused." is sufficient. For example:

```python
def viking_cafe_order(spam: str, beans: str, eggs: str | None = None) -> str:
    del beans, eggs  # Unused by vikings.
    return spam + spam + spam
```

Other common forms of suppressing this warning include using `_` as the identifier for the unused argument, prefixing the argument name with `unused_`, or assigning them to `_`. These forms are allowed but no longer encouraged. These break callers that provide arguments by name and do not enforce that the arguments are actually unused.

## Imports

> 💡 **Tip**: Use `import` statements for packages and modules only, not for individual classes or functions.

**Definition**:
Mechanisms for sharing code from one module to another.

**Pros**:
The namespace management convention is very simple. The source of each identifier is indicated in a consistent way; `x.Obj` says that object `Obj` is defined in module `x`.

**Cons**:
Module names can still collide. Some module names are inconveniently long.

**Decision**:
1. Use `import x` for importing packages and modules.
2. Use `from x import y` where `x` is the package prefix and `y` is the module name with no prefix.
3. Use `from x import y as z` if two modules named `y` are to be imported or if `y` conflicts with a top-level name defined in the current module.
4. Use `import y as z` only when `z` is a standard abbreviation (e.g., `import numpy as np`).

For example the module `sound.effects.echo` may be imported as follows:

```python
from sound.effects import echo
...
echo.EchoFilter(input, output, delay=0.7, atten=4)
```

Do not use relative names in imports. Even if the module is in the same package, use the full package name. This helps prevent unintentionally importing a package twice.

**Exceptions**:
The following are exempt from this rule:
1. Symbols from the following modules are used to support static analysis and type checking:
    - `typing` module
    - `collections.abc` module
    - `typing_extensions` module
2. Redirects from the [six.moves](https://six.readthedocs.io/#module-six.moves) module.

## Packages

> 💡 **Tip**: Import each module using the full path name of the module.

**Pros**:
Avoids conflicts in module names, and avoids imports of incorrect packages due to module search path not being what the author expected. Makes it easier to find modules.

**Cons**:
Makes it harder to deploy code because you have to replicate the package hierarchy. Not really a problem with modern deployment mechanisms.

**Decision**:
All new code should import each module by its full package name.

Imports should happen like this:

Correct:

```python
# Reference the full name absl.flags in the code (verbose).
import absl.flags
from doctor.who import jodie

_FOO = absl.flags.DEFINE_string(...)
```

```python
# Reference the module name flags in the code (common).
from absl import flags
from doctor.who import jodie

_FOO = flags.DEFINE_string(...)
```

Incorrect: (assuming this file is in `doctor/who/` and `jodie.py` also exists there)

```python
# Unclear what is being imported and where it comes from.
# The actual module imported depends on the external environment's sys.path.
# Which jodie module was the author's intent?
import jodie
```

Program directories should not be assumed to be in `sys.path`, even if they sometimes are. Therefore, `import jodie` should be expected to import a third party or top-level package named `jodie`, not a local `jodie.py`.

## Exceptions

> 💡 **Tip**: Exceptions are allowed but must be used with care.

**Definition**:
Exceptions are a means of breaking out of the normal control flow of a code block to handle errors or other exceptional conditions.

**Pros**:
The control flow of normal operation code is not cluttered by error-handling code. It also allows the control flow to skip multiple frames when a certain condition occurs, e.g., returning from N nested functions in one step instead of having to carry-through error codes.

**Cons**:
May cause the control flow to be confusing. Easy to miss error cases when making library calls.

**Decision**:
Exceptions must follow certain conditions:

1. Raise exceptions like this: `raise MyError('Error message')` or `raise MyError()`. Do not use the two-argument form (`raise MyError, 'Error message'`).
2. Make use of built-in exception classes when it makes sense. For example, raise a `ValueError` if you were passed a negative number but expected a positive one. Do not use `assert` statements for validating argument values of a public API. `assert` is used to ensure internal correctness, not to enforce correct usage of the API nor to indicate that something unexpected happened. If an exception is desired in the latter cases, use a raise statement. For example:

    Correct:

    ```python
    def connect_to_next_port(self, minimum: int) -> int:
        """Connects to the next available port.

        Args:
          minimum: A port number at least 1024.

        Returns:
          The new minimum port.

        Raises:
          ConnectionError: If no available port is found.
        """
        if minimum < 1024:
            # Note that this raise of ValueError is not documented because
            # a fix by the caller of the API is required.
            raise ValueError(f'Min. port must be at least 1024, not {minimum}.')
        port = self._find_next_open_port(minimum)
        if port is None:
            raise ConnectionError(
                f'Could not connect to service on port {minimum} or higher.')
        assert port >= minimum, (
            f'Unexpected port {port} less than {minimum}.')
        return port
    ```

    Incorrect:

    ```python
    def connect_to_next_port(self, minimum: int) -> int:
        """Connects to the next available port.

        Args:
          minimum: A port number at least 1024.

        Returns:
          The new minimum port.
        """
        assert minimum >= 1024, 'Min. port must be at least 1024.'
        port = self._find_next_open_port(minimum)
        assert port is not None
        return port
    ```

3. Libraries or packages may define their own exceptions. Those must inherit from an existing exception class. Exception names should end in `Error` and should not introduce repetition (`foo.FooError`).
4. Never use catch-all `except:` statements, or catch `Exception` or `StandardError`, unless you are re-raising the exception or in the outermost block in your thread (and logging an error message). Python is very tolerant in this regard and `except:` will really catch everything including misspelled identifiers, `sys.exit()` calls, `Ctrl+C` interrupts, unit test failures and all kinds of other exceptions you simply don't want to catch.
5. Minimize the amount of code in a `try`/`except` block. The larger the body of the `try`, the more likely that an exception will be raised by a line of code that you didn't expect to raise an exception. In those cases, the `try`/`except` block masks a real error.
6. Use the `finally` clause to execute code whether or not an exception is raised in the `try` block. This is often useful for cleanup, i.e., closing a file.

## Global variables

> 💡 **Tip**: Avoid global variables.

**Definition**:
Variables that are declared at the module level or as class attributes.

**Pros**:
Occasionally useful.

**Cons**:
1. Has the potential to change module behavior during the import, because assignments to global variables are done when the module is first imported.
2. Breaks encapsulation: Can make it hard to achieve some goals. For example, if you use a global variable to manage a database connection, it's hard to connect to two different databases at the same time (such as for comparing differences during a migration). Similar problems exist for global registries.

**Decision**:
Avoid global variables.

In those rare cases where they are needed, module-level variables should be declared as internal to the module and accessed through public functions. See [Naming](#naming) below.

Module-level constants are permitted and encouraged. For example: `_MAX_HOLY_HANDGRENADE_COUNT = 3`. Constants must be named using all caps with underscores. See [Naming](#naming) below.

## Nested/Local/Inner Classes and Functions

> 💡 **Tip**: Local helper functions or classes are allowed when they are used to wrap up a bit of logic that is only used inside a single function.

**Definition**:
A class can be defined inside a method, function, or class. A function can be defined inside a method or function. Nested functions have read-only access to variables defined in enclosing scopes.

**Pros**:
Allows definition of utility classes and functions that are only used inside a very limited scope. Very "Pythonic". Often used for Decorators.

**Cons**:
Nested functions and classes cannot be directly unit tested. Nesting can make your outer function longer and less readable.

**Decision**:
They are fine. Avoid nested functions or classes except when they are closing over a local value other than `self` or `cls`. Do not nest a function just to hide it from users of a module. Instead, prefix its name with an `_` at the module level so that it can still be accessed by tests.

## Comprehensions & Generator Expressions

> 💡 **Tip**: Okay to use for simple cases.

**Definition**:
List, Dict, and Set comprehensions and generator expressions provide a concise and efficient way to create container types and iterators without resorting to the use of `map()`, `filter()`, or `lambda`.

**Pros**:
Simple comprehensions can be clearer and simpler than other dict, list, or set creation techniques. Generator expressions can be very efficient, since they avoid the creation of a list entirely.

**Cons**:
Complicated comprehensions or generator expressions can be hard to read.

**Decision**:
Okay to use for simple cases. Each portion must fit on one line: mapping expression, `for` clause, `filter` expression. Multiple `for` clauses or filter expressions are not allowed. Use looping instead when things get more complicated.

## Default Iterators and Operators

> 💡 **Tip**: Use default iterators and operators for types that support them, like lists, dictionaries, and files.

**Definition**:
Container types like dictionaries and lists have default iterators and membership test operators (`in` and `not in`).

**Pros**:
The default iterators and operators are simple and efficient. They express the operation directly without extra function calls. A function that uses default operators is generic. It can be used with any type that supports the operation.

**Cons**:
You can't tell the type of the objects by reading the method names (e.g. `has_key()`). This is also a pro.

**Decision**:
Use default iterators and operators for types that support them, like lists, dictionaries, and files. The built-in types also define iterator methods. Use those methods instead of methods that return lists, except that you should not change a container while iterating over it.

## Generators

> 💡 **Tip**: Use generators as needed.

**Definition**:
A generator function returns an iterator that yields a value each time it executes a `yield` statement. After yielding a value, the runtime state of the generator function is suspended until the next value is needed.

**Pros**:
Simpler code, because the state of local variables and control flow are preserved for each call. Generators use less memory than functions that create an entire list of values at once.

**Cons**:
None.

**Decision**:
Fine. Use "Yields:" rather than "Returns:" in the docstring for generator functions.

## Lambda Functions

> 💡 **Tip**: Okay for one-liners.

**Definition**:
Lambdas define anonymous functions in an expression, as opposed to a statement.

**Pros**:
Convenient.

**Cons**:
Harder to read and debug than local functions. The lack of a name means stack traces are more difficult to understand. Expressiveness is limited because the function may only contain a single expression.

**Decision**:
Okay for one-liners. If the code inside the lambda function is any longer than 60-80 characters, it's probably better to define it as a regular (nested) function.

## Conditional Expressions

> 💡 **Tip**: Okay for simple cases.

**Definition**:
Conditional expressions (ternary operators) are shorter syntax for if-statements. For example: `x = 1 if cond else 2`.

**Pros**:
Shorter and more convenient than an if-statement.

**Cons**:
May be harder to read than an if-statement. The condition may be difficult to locate if the expression is long.

**Decision**:
Okay for simple cases. Each portion must fit on one line: true-expression, if-condition, else-expression. Use a complete if-statement when things get more complicated.

## Default Argument Values

> 💡 **Tip**: Okay in most cases.

**Definition**:
You can specify values for variables at the end of a function's parameter list, e.g., `def foo(a, b=0):`. If `foo` is called with only one argument, `b` is set to 0. If it is called with two arguments, `b` has the value of the second argument.

**Pros**:
Often you have a function that uses lots of default values, but on rare occasions you want to override the defaults. Default argument values provide an easy way to do this, without having to define lots of functions for the rare exceptions. Also, Python does not support overloaded methods/functions and default arguments are an easy way to "fake" the overloading behavior.

**Cons**:
Default arguments are evaluated once at module load time. This may cause problems if the argument is a mutable object such as a list or a dictionary. If the function modifies the object (e.g., by appending an item to a list), the default value is modified.

**Decision**:
Okay with the following exception:

Do not use mutable objects as default values in the function or method definition.

Correct:

```python
def foo(a, b=None):
    if b is None:
        b = []
```

Incorrect:

```python
def foo(a, b=[]):
    ...
```

## Properties

> 💡 **Tip**: Use properties for accessing or setting data where you would otherwise use simple getter or setter methods.

**Definition**:
A way to wrap function calls for getting and setting an attribute as a standard attribute access.

**Pros**:
Readability is improved by eliminating explicit getter and setter method calls for simple attribute access. Allows calculations to be lazy. Maintains the public interface of a class by allowing the internal implementation to change without changing how the class is used.

**Cons**:
Operators are evaluated once at module load time. Must be used with care as they can hide side-effects, just like operator overloading. Might be confusing for subclasses.

**Decision**:
Use properties in new code to access or set data where you would otherwise use simple getter or setter methods. Properties should be created with the `@property` decorator.

Inheritance with properties can be non-obvious. Do not use properties to implement computation that a subclass might expect to override and extend.

## True/False Evaluations

> 💡 **Tip**: Use the "implicit" false if at all possible.

**Definition**:
Python evaluates certain values as false in a boolean context. A quick rule of thumb is that all "empty" values are considered false, so `0, None, [], {}, ""` all evaluate as false in a boolean context.

**Pros**:
Conditions using Python booleans are easier to read and less error-prone. In most cases, they're also faster.

**Cons**:
May look strange to C/C++ developers.

**Decision**:
Use the "implicit" false if at all possible, e.g., `if foo:` rather than `if foo != []:`. There are a few caveats that you should keep in mind:

1. Always use `if foo is None:` (or `is not None`) to check for a `None` value. e.g., when testing whether a variable or argument that defaults to `None` was set to some other value. The other value might be a value that's false in a boolean context!
2. Never compare a boolean variable to `False` using `==`. Use `if not x:` instead. If you need to distinguish `False` from `None` then use a chain of expressions, such as `if not x and x is not None:`.
3. For sequences (strings, lists, tuples), use the fact that empty sequences are false, so `if not seq:` or `if seq:` is preferable to `if len(seq) == 0:` or `if len(seq) > 0:`.
4. When handling integers, implicit false may involve more risk than benefit (e.g., accidentally handling `None` as 0). You may compare a value which is known to be an integer (and is not the result of `len()`) against the integer 0.

## Lexical Scoping

> 💡 **Tip**: Okay to use.

**Definition**:
A nested Python function can refer to variables defined in enclosing functions, but cannot assign to them. Variable bindings are resolved using lexical scoping, that is, based on the static program text. Any assignment to a name in a block will cause Python to treat all references to that name as a local variable, even if the use occurs before the assignment. If a global declaration occurs, then the name is treated as a global variable.

**Pros**:
Often results in clearer, more elegant code. Especially comforting to experienced Lisp and Scheme (and Haskell, and ML, and ...) programmers.

**Cons**:
Can lead to confusing bugs.

**Decision**:
Okay to use.

## Function and Method Decorators

> 💡 **Tip**: Use decorators judiciously when there is a clear advantage. Avoid `staticmethod`. Use `classmethod` sparingly.

**Definition**:
Decorators for functions and methods (the `@` notation). One common decorator is `@property`, used for converting ordinary methods into dynamically computed attributes. However, the decorator syntax allows for user-defined decorators as well.

**Pros**:
Elegantly specifies some transformation on a method; the transformation might otherwise eliminate some set of repetitive code, enforce invariants, etc.

**Cons**:
Decorators can perform arbitrary operations on a function's arguments or return values, potentially leading to surprising implicit behavior. Moreover, decorators are executed at import time. Failures in decorator code are pretty much impossible to recover from.

**Decision**:
Use decorators judiciously when there is a clear advantage. Decorators should follow the same importing and naming guidelines as functions. Decorator pydoc should clearly state that the function is a decorator. Write unit tests for decorators.

Avoid external dependencies in the decorator itself (e.g. don't rely on files, sockets, database connections, etc.), since they might not be available when the decorator runs (at import time, perhaps from `pydoc` or other tools). A decorator that is called with valid arguments should (as much as possible) be guaranteed to succeed.

Decorators are a special case of "top-level code" - see main for more discussion.

Never use `staticmethod` unless forced to in order to integrate with an API defined in an existing library. Write a module-level function instead.

Use `classmethod` only when writing a named constructor or a class-specific routine that manages important class-wide state such as a process-wide cache.

## Threads

> 💡 **Tip**: Do not rely on the atomicity of built-in types.

While Python's built-in types such as dictionaries appear to have atomic operations, there are corner cases where they don't (e.g. if `__hash__` or `__eq__` are implemented as Python methods) and their atomicity should not be relied upon. Neither should you rely on atomic variable assignment (since this in turn depends on dictionaries).

Use the `queue` module's `Queue` data type as the preferred way to communicate data between threads. Otherwise, use the `threading` module and its locking primitives. Learn about the proper usage of condition variables using `threading.Condition` instead of using lower-level locks.

## Power Features

> 💡 **Tip**: Avoid these features.

**Definition**:
Python is an extremely flexible language and gives you many fancy features such as custom metaclasses, access to bytecode, on-the-fly compilation, dynamic inheritance, object reparenting, import hacks, reflection (e.g. `getattr()`), modification of system internals, etc.

**Pros**:
These are powerful language features. They can make your code more compact.

**Cons**:
It's very tempting to use these "cool" features when they're not absolutely necessary. They're hard to read, understand, and debug. It might seem like a good idea in the beginning, but when you come back to the code, it's often much harder to understand than longer but straightforward code.

**Decision**:
Avoid these features in your code.

Standard library modules and classes that internally use these features are okay to use (for example, `abc.ABCMeta`, `dataclasses`, and `enum`).

## Modern Python: from \_\_future\_\_ imports

> 💡 **Tip**: Being able to use syntax from future versions of Python is provided by the `from __future__ import` statement.

**Definition**:
Using `from __future__ import` and the modern syntax allowed by it helps you use newer Python features today.

**Pros**:
Predictably, it has been shown to make the upgrade process smoother since it can be done on a file-by-file basis while the compatibility statement protects against regressions. Modern code is easier to maintain since it's less likely to accumulate tech debt that gets in the way of a runtime upgrade.

**Decision**:
The use of `from __future__ import` statements is encouraged.

## Type Annotated Code

> 💡 **Tip**: You can annotate Python 3 code with type hints according to [PEP-484](https://www.python.org/dev/peps/pep-0484/).

**Definition**:
Type annotations (or "type hints") are for function or method arguments and return values:

```python
def func(a: int) -> list[int]:
```

You can also declare the type of a variable using [PEP-526](https://www.python.org/dev/peps/pep-0526/) syntax:

```python
a: SomeType = some_func()
```

**Pros**:
Type annotations improve the readability and maintainability of your code. Static type checkers can catch many common errors and suggest better ways to structure your code.

**Cons**:
You have to keep the type declarations up to date. You may see false positives from the type checker.

**Decision**:
You are strongly encouraged to use Python type analysis when updating code. When adding or modifying public APIs, use type annotations and enable checking via pytype in the build system.
