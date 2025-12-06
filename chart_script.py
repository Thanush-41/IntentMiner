
import plotly.graph_objects as go

# Create figure
fig = go.Figure()

# Define node positions for the flowchart
# Format: (x, y, label, type) where type is 'stage', 'decision', 'start', or 'end'
nodes = [
    (0.5, 10, 'Start', 'start'),
    (0.5, 9, 'Stage 1:<br>Data Ingestion', 'stage'),
    (0.5, 8, 'Stage 2:<br>Preprocessing', 'stage'),
    (0.5, 7, 'Guardrail:<br>Quality Check', 'decision'),
    (0.5, 6, 'Stage 3:<br>Intent Extract', 'stage'),
    (0.5, 5, 'Stage 4:<br>Classification', 'stage'),
    (0.5, 4, 'Guardrail:<br>Confidence', 'decision'),
    (0.2, 3, 'Stage 5:<br>Refinement', 'stage'),
    (0.5, 2, 'Stage 6:<br>Validation', 'stage'),
    (0.5, 1, 'Guardrail:<br>Final Check', 'decision'),
    (0.5, 0, 'Stage 7:<br>Output', 'stage'),
    (0.5, -1, 'End', 'end'),
]

# Add nodes as shapes and text
for x, y, label, node_type in nodes:
    if node_type == 'stage':
        # Rectangle for stages
        fig.add_shape(type="rect", x0=x-0.12, y0=y-0.3, x1=x+0.12, y1=y+0.3,
                     line=dict(color="#21808d", width=2), fillcolor="#B3E5EC")
    elif node_type == 'decision':
        # Diamond for decision/guardrails
        fig.add_shape(type="path",
                     path=f"M {x},{y+0.3} L {x+0.12},{y} L {x},{y-0.3} L {x-0.12},{y} Z",
                     line=dict(color="#21808d", width=2), fillcolor="#FFEB8A")
    elif node_type in ['start', 'end']:
        # Rounded rectangle for start/end
        fig.add_shape(type="rect", x0=x-0.1, y0=y-0.25, x1=x+0.1, y1=y+0.25,
                     line=dict(color="#21808d", width=2), fillcolor="#A5D6A7")
    
    fig.add_annotation(x=x, y=y, text=label, showarrow=False,
                      font=dict(size=10, color="#13343b"))

# Add arrows for data flow
arrows = [
    (0.5, 9.7, 0.5, 9.3),  # Start -> Stage 1
    (0.5, 8.7, 0.5, 8.3),  # Stage 1 -> Stage 2
    (0.5, 7.7, 0.5, 7.3),  # Stage 2 -> Guard 1
    (0.5, 6.7, 0.5, 6.3),  # Guard 1 -> Stage 3 (Pass)
    (0.5, 5.7, 0.5, 5.3),  # Stage 3 -> Stage 4
    (0.5, 4.7, 0.5, 4.3),  # Stage 4 -> Guard 2
    (0.4, 3.7, 0.25, 3.3),  # Guard 2 -> Stage 5 (Low)
    (0.6, 4, 0.75, 2.3),  # Guard 2 -> Stage 6 (High)
    (0.2, 2.7, 0.35, 2.3),  # Stage 5 -> Stage 6
    (0.5, 1.7, 0.5, 1.3),  # Stage 6 -> Guard 3
    (0.5, 0.7, 0.5, 0.3),  # Guard 3 -> Stage 7 (Pass)
    (0.5, -0.3, 0.5, -0.75),  # Stage 7 -> End
]

# Feedback arrows
feedback = [
    (0.4, 7, 0.3, 8, 'Fail'),  # Guard 1 -> Stage 2 (Fail)
    (0.35, 1, 0.15, 6, 'Fail'),  # Guard 3 -> Stage 3 (Fail)
]

for x0, y0, x1, y1 in arrows:
    fig.add_annotation(x=x1, y=y1, ax=x0, ay=y0, 
                      xref='x', yref='y', axref='x', ayref='y',
                      showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=2,
                      arrowcolor="#21808d")

# Add feedback arrows with labels
for x0, y0, x1, y1, label in feedback:
    fig.add_annotation(x=x1, y=y1, ax=x0, ay=y0,
                      xref='x', yref='y', axref='x', ayref='y',
                      showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=2,
                      arrowcolor="#DB4545")

# Add edge labels
fig.add_annotation(x=0.52, y=6.5, text="Pass", showarrow=False,
                  font=dict(size=9, color="#13343b"))
fig.add_annotation(x=0.32, y=7.5, text="Fail", showarrow=False,
                  font=dict(size=9, color="#DB4545"))
fig.add_annotation(x=0.3, y=3.5, text="Low", showarrow=False,
                  font=dict(size=9, color="#13343b"))
fig.add_annotation(x=0.7, y=3.2, text="High", showarrow=False,
                  font=dict(size=9, color="#13343b"))
fig.add_annotation(x=0.52, y=0.5, text="Pass", showarrow=False,
                  font=dict(size=9, color="#13343b"))
fig.add_annotation(x=0.25, y=3.5, text="Fail", showarrow=False,
                  font=dict(size=9, color="#DB4545"))

# Update layout
fig.update_layout(
    title="IntentMiner 7-Stage Pipeline",
    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-0.1, 1.1]),
    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-1.5, 10.5]),
    plot_bgcolor='#F3F3EE',
    showlegend=False
)

# Save the figure
fig.write_image('intentminer_pipeline.png')
fig.write_image('intentminer_pipeline.svg', format='svg')

print("IntentMiner pipeline flowchart created successfully!")
