import os
def log_values(cost, grad_norms, epoch, batch_id, step,
               log_likelihood, reinforce_loss, bl_loss, writer, opts):
    avg_cost = cost.mean().item()
    grad_norms, grad_norms_clipped = grad_norms

    # Log values to screen
    print('epoch: {}, train_batch_id: {}, avg_cost: {}'.format(epoch, batch_id, avg_cost))

    print('grad_norm: {}, clipped: {}'.format(grad_norms[0], grad_norms_clipped[0]))
    log_dir = os.path.join(
        'logs',
        "{}_{}".format(opts.problem, opts.graph_size),
        opts.run_name
    )
    with open(os.path.join(log_dir, 'trace.txt'), 'a+') as f:
        f.writelines('epoch: {}, train_batch_id: {}, avg_cost: {}'.format(epoch, batch_id, avg_cost)+ '\r\n')
    # Log values to tensorboard
    if not opts.no_tensorboard:
        writer.add_scalar('avg_cost', avg_cost, step)

        writer.add_scalar('actor_loss', reinforce_loss.item(), step)
        writer.add_scalar('nll', -log_likelihood.mean().item(), step)

        writer.add_scalar('grad_norm', grad_norms[0], step)
        writer.add_scalar('grad_norm_clipped', grad_norms_clipped[0], step)

        if opts.baseline == 'critic':
            writer.add_scalar('critic_loss', bl_loss.item(), step)
            writer.add_scalar('critic_grad_norm', grad_norms[1], step)
            writer.add_scalar('critic_grad_norm_clipped', grad_norms_clipped[1], step)
