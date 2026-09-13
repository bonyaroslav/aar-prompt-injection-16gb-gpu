# Start here

This study asks whether one prompt-injection fine-tuning recipe could meet
preset requirements on one RTX 4080 with 16 GB GPU memory. It produced nine
checkpoints across three runs. All failed the capability requirements, so none
was selected. The result is specific to this recipe and evaluation setup.

## Choose a reading route

| What you want to know | Read |
|---|---|
| Why I ran this, and what I got wrong | [Personal retrospective](retrospective.md) |
| What was tested, and what happened? | [Checked fact sheet](publication/direction-a/fact-sheet.md) |
| What do the injection scores actually establish? | [Scoring audit](publication/direction-a/scorer-fact-sheet.md) |
| Can I check the arithmetic myself? | [Reproduction walkthrough](publication/direction-a/README.md) |
| How does this connect to earlier research? | [Anthropic and SecFid references](../README.md#scientific-foundations) |
| What is the fuller argument and its limitations? | [Directions A/B knowledge base](publication-directions-a-b-knowledge-base.md) |
| What rules were fixed for the experiment? | [Protocol](../protocol/manifest.json) |

The numeric reanalysis requires Python and no GPU. Original training and model
evaluation are separate, substantially larger tasks. Raw generated answers were
not retained, so these records cannot establish a behavioral cause for the changes.

Return to the [repository overview](../README.md).
