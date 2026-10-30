"""ESA main-API variant integration and regression tests; CPU only."""
import json
import tempfile
from pathlib import Path

import pytest
import torch

from esa import ESA, ESAConfig, ESAModel, ESAModelConfig
from esa.backends.thunder import ThunderESA
from esa.backends.swqr_enhanced import EnhancedSWQRLayer
from esa.swqr_trainable import SWQRConfig, SWQRLayer


def config(variant=None, layers=2):
    kw = dict(vocab_size=64, block=256, n_layer=layers, head=6,
              embd=48, dropout=0.0, precision='fp32', compute_dtype='fp32',
              backend='pytorch', training_compile=False)
    if variant is not None:
        kw['variant'] = variant
    return ESAModelConfig(**kw)


def test_default_vs_base_and_api():
    x=torch.randn(2,13,48)
    a=ESA(embd=48,head=6,device='cpu',precision='fp32')
    b=ESA(embd=48,head=6,device='cpu',precision='fp32',variant='base')
    assert a.variant == 'enhanced'
    assert isinstance(a.layer,EnhancedSWQRLayer)
    assert isinstance(b.layer,ThunderESA)
    assert a(x).shape==b(x).shape==x.shape
    assert ESAConfig(48).variant=='enhanced'
    assert config().variant=='enhanced'
    with pytest.raises(ValueError): ESA(48,variant='wrong',device='cpu')
    with pytest.raises(ValueError): ESAConfig(48,variant='wrong')
    with pytest.raises(ValueError): ESAModelConfig(vocab_size=64,variant='wrong')


def test_original_base_weights_and_outputs_unchanged():
    torch.manual_seed(4)
    layer=ESA(48,head=6,device='cpu',precision='fp32',variant='base',backend='pytorch')
    torch.manual_seed(4)
    reference=ThunderESA(embd=48,head=6,precision='fp32',runtime_backend='pytorch')
    assert set(reference.state_dict())==set(layer.layer.state_dict())
    for k,v in reference.state_dict().items():
        assert torch.equal(v,layer.layer.state_dict()[k]),k
    layer.eval();reference.eval()
    x=torch.randn(2,67,48)
    with torch.no_grad():
        assert torch.allclose(layer(x),reference(x),atol=0,rtol=0)


@pytest.mark.parametrize('length', [1,17,128,129,256,257])
def test_enhanced_pack_matches_original_swqr(length):
    torch.manual_seed(91)
    mixer=ESA(48,head=6,precision='fp32',device='cpu',variant='enhanced')
    ref=SWQRLayer(SWQRConfig(dim=48,heads=6,layers=1,chunk_size=32))
    ref.load_state_dict(mixer.layer.state_dict(),strict=True)
    mixer.eval()
    x=torch.randn(2,length,48)
    state=torch.randn(2,6,8)
    with torch.inference_mode():
        y,s=mixer.prefill(x,state)
        ry,rs=ref(x,state)
    torch.testing.assert_close(y,ry,atol=2e-5,rtol=2e-5)
    torch.testing.assert_close(s,rs,atol=2e-5,rtol=2e-5)
    assert not any('packed' in name for name in mixer.state_dict())


def test_mixer_decode_input_and_state_layouts():
    e=ESA(48,head=6,device='cpu').eval()
    for inp in (torch.randn(2,48),torch.randn(2,1,48)):
        for layout in ('heads','flat'):
            state=e.init_state(2,layout=layout)
            with torch.no_grad(): y,s=e.decode_step(inp,state)
            assert y.shape==inp.shape
            assert s.shape==state.shape


def test_gradients_and_pack_refresh_after_optimizer_and_state_load():
    torch.manual_seed(42)
    e=ESA(48,head=6,device='cpu',variant='enhanced')
    x=torch.randn(2,19,48)
    e.train()
    loss=e(x).square().mean()
    loss.backward()
    assert e.layer.W_A.weight.grad is not None
    assert e.layer.write_up.weight.grad is not None
    assert e.layer.read_up.weight.grad is not None
    opt=torch.optim.SGD(e.parameters(),lr=0.01)
    opt.step()
    e.eval()
    with torch.no_grad(): y_before=e(x)
    assert e.layer._packs_valid
    sd={k:v.clone() for k,v in e.state_dict().items()}
    sd['layer.W_V.weight'].add_(0.2)
    e.load_state_dict(sd)
    assert not e.layer._packs_valid
    with torch.inference_mode(): y_after=e(x)
    assert e.layer._packs_valid
    assert not torch.allclose(y_before,y_after)
    expected=SWQRLayer(SWQRConfig(dim=48,heads=6,layers=1,chunk_size=32))
    expected.load_state_dict(e.layer.state_dict())
    with torch.no_grad(): y_ref,_=expected(x)
    torch.testing.assert_close(y_after,y_ref,atol=2e-5,rtol=2e-5)


@pytest.mark.parametrize('variant',['base','enhanced'])
def test_complete_model_train_prefill_decode_generate_and_save(variant):
    torch.manual_seed(171)
    model=ESAModel(config(variant),device='cpu')
    ids=torch.randint(0,64,(1,17))
    targets=torch.randint(0,64,(1,17))
    logits,loss=model(ids,targets)
    assert logits.shape==(1,17,64)
    assert loss.isfinite()
    loss.backward()
    assert any(p.grad is not None for p in model.blocks[0].esa.parameters())
    model.eval()
    with torch.inference_mode():
        x,state,n=model.prefill(ids)
        assert x.shape==(1,64) and state.shape==(2,1,6,8) and n==17
        next_logits,next_state=model.lightning_step(ids[:,-1],state,torch.tensor([17]))
        assert next_logits.shape==(1,1,64) and next_state.shape==state.shape
        output=model.generate(input_ids=ids[:,:5],max_new_tokens=3,compile=False)
        assert output.shape==(1,8)
    with tempfile.TemporaryDirectory() as tmp:
        model.save(tmp)
        saved=json.loads((Path(tmp)/'config.json').read_text())
        assert saved['variant']==variant
        other=ESAModel.load(tmp,device='cpu')
        assert other.config.variant==variant
        other.eval()
        with torch.inference_mode():
            y2,s2,n2=other.prefill(ids)
        torch.testing.assert_close(x,y2,rtol=0,atol=0)
        torch.testing.assert_close(state,s2,rtol=0,atol=0)
        assert n2==n


def test_legacy_checkpoint_without_variant_loads_as_base():
    model=ESAModel(config('base'),device='cpu').eval()
    with tempfile.TemporaryDirectory() as tmp:
        model.save(tmp)
        path=Path(tmp)/'config.json'
        cfg=json.loads(path.read_text())
        cfg.pop('variant')
        path.write_text(json.dumps(cfg))
        loaded=ESAModel.load(tmp,device='cpu')
        assert loaded.config.variant=='base'
        assert all(block.esa.variant=='base' for block in loaded.blocks)
        assert ESAModelConfig.from_dict(cfg).variant=='base'


def test_unsupported_native_enhanced_backend_rejected():
    with pytest.raises(ValueError,match='native'):
        ESA(48,head=6,device='cpu',variant='enhanced',backend='native')
    e=ESA(48,head=6,device='cpu',variant='enhanced')
    with pytest.raises(ValueError,match='native'):
        e.set_backend('native')
    with pytest.raises(ValueError,match='native'):
        ESAModel(config('enhanced'),device='cpu').set_backend('native')

@pytest.mark.parametrize('variant', ['base','enhanced'])
def test_prefill_forward_and_decode_consistency(variant):
    torch.manual_seed(310)
    m=ESAModel(config(variant),device='cpu').eval()
    ids=torch.randint(0,64,(1,31))
    with torch.inference_mode():
        full,_=m(ids)
        pre,s,n=m.prefill(ids)
        earlier,old_states,n2=m.prefill(ids[:,:-1])
        decoded,new_states=m.lightning_step(ids[:,-1],old_states,torch.tensor([30]))
    torch.testing.assert_close(full[:,-1],pre,atol=1e-5,rtol=1e-5)
    torch.testing.assert_close(full[:,-1],decoded[:,0],atol=1e-5,rtol=1e-5)
    torch.testing.assert_close(new_states,s,atol=1e-5,rtol=1e-5)


def test_compiled_mixer_forward_prefill_decode_dispatch():
    if not hasattr(torch,'compile'):
        pytest.skip('torch.compile not installed')
    e=ESA(48,head=6,device='cpu',variant='enhanced').eval()
    e.layer=torch.compile(e.layer,backend='eager',fullgraph=True,dynamic=False)
    x=torch.randn(1,17,48)
    with torch.inference_mode():
        y=e(x)
        f,state=e.prefill(x)
        d,s=e.decode_step(x[:,:1],state)
    torch.testing.assert_close(y,f,atol=1e-5,rtol=1e-5)
    assert d.shape==(1,1,48) and s.shape==state.shape
