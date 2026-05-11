# Documento 2: Políticas internas y procedimientos de NovaTech Solutions

## Horario laboral y modalidad de trabajo

NovaTech Solutions aplica una política de trabajo híbrido pensada para equilibrar productividad, colaboración y bienestar. El horario general de referencia es de lunes a viernes, de 9:00 a 18:00, con una pausa flexible para comer entre las 13:30 y las 15:30. La empresa permite adaptar la entrada entre las 8:00 y las 10:00, siempre que se cumplan los compromisos del equipo y se mantenga disponibilidad durante la franja común obligatoria, que va de 10:00 a 13:30 y de 15:30 a 17:00. En esa franja se programan reuniones, revisiones de proyecto y sesiones con clientes.

La modalidad híbrida establece un mínimo de dos días presenciales a la semana para equipos de desarrollo, consultoría y operaciones. Los equipos pueden elegir los días presenciales, aunque se recomienda que cada proyecto tenga al menos una jornada común para facilitar coordinación. El teletrabajo completo puede autorizarse de forma temporal por razones justificadas, como desplazamientos, necesidades familiares, enfermedad leve o concentración en tareas que requieren trabajo profundo. Cualquier cambio prolongado de modalidad debe comunicarse al responsable directo y quedar registrado en la herramienta interna de planificación.

## Política de vacaciones y ausencias

Cada empleado dispone de veintitrés días laborables de vacaciones al año, además de los festivos oficiales aplicables en su ubicación. Las vacaciones deben solicitarse con al menos quince días naturales de antelación, salvo casos urgentes. La aprobación depende de la disponibilidad del equipo, la continuidad de los proyectos y los compromisos con clientes. Para evitar bloqueos operativos, no se recomienda que más del cuarenta por ciento de un mismo equipo esté de vacaciones simultáneamente.

Las ausencias por enfermedad deben comunicarse lo antes posible al responsable directo. Si la ausencia supera tres días naturales, se debe aportar la documentación correspondiente según la normativa aplicable. Para asuntos personales breves, la empresa permite solicitar permisos puntuales, que se revisan caso por caso. NovaTech Solutions promueve que los empleados avisen con transparencia y antelación siempre que sea posible.

## Beneficios internos

NovaTech Solutions ofrece varios beneficios orientados al aprendizaje, la salud y la conciliación. Cada empleado cuenta con un presupuesto anual ficticio de 900 euros para formación profesional, que puede utilizarse en cursos, certificaciones, libros técnicos o asistencia a conferencias relacionadas con su puesto. Para utilizar este presupuesto, el empleado debe presentar una breve justificación indicando cómo la formación contribuirá a sus objetivos profesionales y al trabajo del equipo.

La empresa también ofrece sesiones internas de intercambio de conocimiento cada dos viernes. En estas sesiones, un miembro del equipo presenta un proyecto, una herramienta, una lección aprendida o una buena práctica. Además, existe un programa de mentoría voluntaria para empleados junior, que permite recibir acompañamiento técnico y orientación profesional durante los primeros seis meses.

En materia de bienestar, NovaTech Solutions fomenta pausas saludables, reuniones con duración limitada y desconexión digital fuera del horario laboral. Salvo incidencias críticas previamente definidas, no se espera que los empleados respondan mensajes por la noche, fines de semana o vacaciones. Los responsables deben planificar el trabajo para evitar urgencias artificiales y proteger la concentración del equipo.

## Código de conducta

El código de conducta de NovaTech Solutions se basa en respeto, honestidad, confidencialidad y responsabilidad profesional. Todos los empleados deben tratar a compañeros, clientes y proveedores con cortesía, evitando comentarios discriminatorios, presión indebida o comportamientos que generen un entorno hostil. Las discrepancias técnicas deben resolverse mediante argumentos, pruebas y revisión colaborativa, no mediante ataques personales.

La confidencialidad es especialmente importante porque la empresa trabaja con documentación privada, datos internos y prototipos de inteligencia artificial. Ningún empleado debe copiar información de clientes en herramientas no autorizadas, subir claves a repositorios públicos o compartir documentación sensible fuera de los canales definidos. Las claves de API, contraseñas y tokens deben guardarse en gestores seguros o variables de entorno, nunca dentro del código fuente.

## Procedimiento para proyectos de IA

Todo proyecto de inteligencia artificial debe comenzar con una fase de definición. En esta fase se identifican el problema, los usuarios, las fuentes de datos, los riesgos y los criterios de éxito. Después se realiza una prueba de concepto limitada, donde se valida si la solución recupera información correcta, responde con coherencia y evita inventar datos. En proyectos RAG, el equipo debe documentar la estrategia de chunking, el modelo de embeddings utilizado, el número de fragmentos recuperados, el umbral de similitud si existe y las limitaciones detectadas.

Antes de pasar a producción, se debe realizar una revisión de seguridad y calidad. Esta revisión comprueba que no haya credenciales expuestas, que las respuestas incluyan límites claros, que exista supervisión humana y que los usuarios sepan cuándo el sistema puede equivocarse. Tras el despliegue, el proyecto debe monitorizarse periódicamente para detectar preguntas sin respuesta, fragmentos mal recuperados, errores de interpretación o necesidades de actualización documental.
